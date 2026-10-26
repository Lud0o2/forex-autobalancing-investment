"""Weekly research backtest engine built on the portfolio accounting layer."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from datetime import date, timedelta
from typing import Any

import pandas as pd

from forex_v1.backtest.config import BacktestConfig
from forex_v1.backtest.data import MarketData
from forex_v1.contracts import TargetWeightProvider
from forex_v1.portfolio import (
    PortfolioState,
    create_portfolio,
    mark_to_market,
    value_portfolio,
)
from forex_v1.portfolio.constraints import validate_target_weights
from forex_v1.portfolio.rebalancing import RebalanceResult, rebalance_portfolio


@dataclass(frozen=True, slots=True)
class BacktestExecution:
    """One weekly allocation decision and its later simulated execution."""

    signal_date: date
    information_cutoff_date: date
    information_market_date: date
    execution_date: date
    target_weights: dict[str, float]
    rebalance: RebalanceResult

    @property
    def market_date(self) -> date:
        """Legacy alias for the market observation used to construct the signal."""
        return self.information_market_date


@dataclass(frozen=True, slots=True)
class BacktestDecision:
    """Audit record for each Friday, including decisions that could not execute."""

    signal_date: date
    information_cutoff_date: date
    information_market_date: date | None
    target_weights: dict[str, float] | None
    execution_date: date | None
    status: str


@dataclass(frozen=True, slots=True)
class BacktestDailyRecord:
    """End-of-observation portfolio snapshot after any scheduled execution."""

    market_date: date
    nav_eur: float
    cash_eur: float
    units: Mapping[str, float]
    values_eur: Mapping[str, float]
    weights: Mapping[str, float]


@dataclass(frozen=True, slots=True)
class BacktestResult:
    """Complete daily valuation, decision, and execution history."""

    config: BacktestConfig
    initial_state: PortfolioState
    final_state: PortfolioState
    decisions: tuple[BacktestDecision, ...]
    executions: tuple[BacktestExecution, ...]
    daily_records: tuple[BacktestDailyRecord, ...]
    nav_series: pd.Series

    @property
    def total_fees_eur(self) -> float:
        return float(sum(execution.rebalance.operation_fees_eur for execution in self.executions))

    @property
    def initial_nav_eur(self) -> float:
        if not self.daily_records:
            return float(value_portfolio(self.initial_state).nav_eur)
        return self.daily_records[0].nav_eur

    @property
    def final_nav_eur(self) -> float:
        return float(value_portfolio(self.final_state).nav_eur)

    @property
    def total_return(self) -> float:
        if self.initial_nav_eur == 0:
            raise ZeroDivisionError("Initial NAV must be positive")
        return self.final_nav_eur / self.initial_nav_eur - 1.0


def build_history_frame(
    market_data: MarketData,
    up_to_date: date,
) -> pd.DataFrame:
    """Build a history table that contains no market observation after ``up_to_date``."""

    rows: list[dict[str, Any]] = []
    for market_date in market_data.dates:
        if market_date > up_to_date:
            break
        price_row = market_data.prices_by_date[market_date]
        feature_row = market_data.features_by_date.get(market_date, {})
        for currency, price in sorted(price_row.items()):
            record: dict[str, Any] = {
                "market_date": market_date,
                "currency": currency,
                "eur_per_unit": float(price),
                "available_at": None,
                "availability_known": False,
            }
            for feature_name, value in feature_row.items():
                if feature_name.startswith(f"{currency}__"):
                    record[feature_name.split("__", 1)[1]] = value
            rows.append(record)
    if not rows:
        return pd.DataFrame(
            columns=[
                "market_date",
                "currency",
                "eur_per_unit",
                "available_at",
                "availability_known",
            ]
        )
    return pd.DataFrame(rows).sort_values(["market_date", "currency"]).reset_index(drop=True)


def generate_signal_dates(
    config: BacktestConfig,
    market_data: MarketData,
) -> tuple[date, ...]:
    """Generate every calendar Friday in the configured period, including market holidays."""

    first_market_date = market_data.dates[0]
    last_market_date = market_data.dates[-1]
    start = max(config.start_date or first_market_date, first_market_date)
    end = config.end_date or last_market_date
    if start > end:
        return ()
    first_friday = start + timedelta(days=(config.signal_weekday - start.weekday()) % 7)
    signals: list[date] = []
    while first_friday <= end:
        signals.append(first_friday)
        first_friday += timedelta(days=7)
    return tuple(signals)


def _as_of_market_date(
    signal_date: date,
    market_dates: tuple[date, ...],
    availability_lag_days: int,
) -> date | None:
    cutoff = signal_date - timedelta(days=availability_lag_days)
    eligible = [market_date for market_date in market_dates if market_date <= cutoff]
    return eligible[-1] if eligible else None


def _state_to_mapping(state: PortfolioState) -> dict[str, Any]:
    return {
        "cash_eur": state.cash_eur,
        "units": dict(state.units),
        "prices_eur": dict(state.prices_eur),
        "initial_capital_eur": state.initial_capital_eur,
        "cumulative_fees_eur": state.cumulative_fees_eur,
    }


def run_backtest(
    market_data: MarketData,
    provider: TargetWeightProvider,
    config: BacktestConfig,
    *,
    initial_state: PortfolioState | None = None,
) -> BacktestResult:
    """Simulate daily valuations, Friday signals, and next-observation execution.

    Signal history and the provider's marked current state use only prices dated no
    later than ``signal_date - availability_lag_days``. Execution prices are used
    only after the target weights have been produced.
    """

    market_dates = market_data.dates
    simulation_start = max(config.start_date or market_dates[0], market_dates[0])
    simulation_end = min(config.end_date or market_dates[-1], market_dates[-1])
    simulation_dates = tuple(
        market_date for market_date in market_dates if simulation_start <= market_date <= simulation_end
    )
    if not simulation_dates:
        raise ValueError("No market observations fall within the configured backtest period")

    if initial_state is None:
        initial_state = create_portfolio(
            config.initial_capital_eur,
            market_data.prices_by_date[simulation_dates[0]],
        )
    state = initial_state
    signals = generate_signal_dates(config, market_data)
    signal_index = 0
    decisions: list[BacktestDecision] = []
    executions: list[BacktestExecution] = []
    scheduled: dict[date, tuple[BacktestDecision, dict[str, float]]] = {}
    daily_records: list[BacktestDailyRecord] = []
    def consider_signal(signal_date: date) -> None:
        cutoff = signal_date - timedelta(days=config.availability_lag_days)
        information_date = _as_of_market_date(
            signal_date,
            market_dates,
            config.availability_lag_days,
        )
        if information_date is None or information_date > cutoff:
            decisions.append(
                BacktestDecision(
                    signal_date=signal_date,
                    information_cutoff_date=cutoff,
                    information_market_date=None,
                    target_weights=None,
                    execution_date=None,
                    status="no_eligible_observation",
                )
            )
            return

        execution_date = next(
            (
                market_date
                for market_date in simulation_dates
                if market_date > signal_date
            ),
            None,
        )
        if execution_date is not None and execution_date in scheduled:
            decisions.append(
                BacktestDecision(
                    signal_date=signal_date,
                    information_cutoff_date=cutoff,
                    information_market_date=information_date,
                    target_weights=None,
                    execution_date=None,
                    status="execution_collision",
                )
            )
            return

        information_state = mark_to_market(
            state,
            market_data.prices_by_date[information_date],
        )
        history = build_history_frame(market_data, information_date)
        target_weights = validate_target_weights(
            provider.get_target_weights(history, _state_to_mapping(information_state))
        )
        status = "scheduled" if execution_date is not None else "no_execution_observation"
        decision = BacktestDecision(
            signal_date=signal_date,
            information_cutoff_date=cutoff,
            information_market_date=information_date,
            target_weights=dict(target_weights),
            execution_date=execution_date,
            status=status,
        )
        decisions.append(decision)
        if execution_date is not None:
            scheduled[execution_date] = (decision, dict(target_weights))

    for market_date in simulation_dates:
        while signal_index < len(signals) and signals[signal_index] <= market_date:
            consider_signal(signals[signal_index])
            signal_index += 1

        state = mark_to_market(state, market_data.prices_by_date[market_date])
        scheduled_item = scheduled.pop(market_date, None)
        if scheduled_item is not None:
            decision, target_weights = scheduled_item
            if decision.information_market_date is None:
                raise ArithmeticError("Scheduled decision is missing its information market date")
            rebalance = rebalance_portfolio(
                state,
                target_weights,
                fee_rate=config.transaction_cost_bps / 10_000.0,
            )
            state = rebalance.state
            executions.append(
                BacktestExecution(
                    signal_date=decision.signal_date,
                    information_cutoff_date=decision.information_cutoff_date,
                    information_market_date=decision.information_market_date,
                    execution_date=market_date,
                    target_weights=target_weights,
                    rebalance=rebalance,
                )
            )
            decision_index = decisions.index(decision)
            decisions[decision_index] = BacktestDecision(
                signal_date=decision.signal_date,
                information_cutoff_date=decision.information_cutoff_date,
                information_market_date=decision.information_market_date,
                target_weights=decision.target_weights,
                execution_date=market_date,
                status="executed",
            )

        valuation = value_portfolio(state)
        daily_records.append(
            BacktestDailyRecord(
                market_date=market_date,
                nav_eur=valuation.nav_eur,
                cash_eur=state.cash_eur,
                units=dict(state.units),
                values_eur=dict(valuation.values_eur),
                weights=dict(valuation.weights),
            )
        )

    while signal_index < len(signals):
        consider_signal(signals[signal_index])
        signal_index += 1

    nav_index = pd.Index(
        [record.market_date for record in daily_records],
        name="market_date",
    )
    nav_values = [record.nav_eur for record in daily_records]
    return BacktestResult(
        config=config,
        initial_state=initial_state,
        final_state=state,
        decisions=tuple(decisions),
        executions=tuple(executions),
        daily_records=tuple(daily_records),
        nav_series=pd.Series(nav_values, index=nav_index, name="nav_eur"),
    )


__all__ = [
    "BacktestDailyRecord",
    "BacktestDecision",
    "BacktestExecution",
    "BacktestResult",
    "build_history_frame",
    "generate_signal_dates",
    "run_backtest",
]
