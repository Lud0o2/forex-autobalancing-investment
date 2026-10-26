from __future__ import annotations

from datetime import date, timedelta

import pandas as pd
import pytest

from forex_v1.backtest.config import BacktestConfig, BacktestConfigError
from forex_v1.backtest.data import MarketData, market_data_from_records
from forex_v1.backtest.engine import run_backtest
from forex_v1.backtest.reporting import write_backtest_outputs
from forex_v1.backtest.strategy import FixedWeightProvider
from forex_v1.data.schemas import SUPPORTED_CURRENCIES


def prices(usd: float = 0.8, jpy: float = 0.006) -> dict[str, float]:
    return {
        "EUR": 1.0,
        "USD": usd,
        "GBP": 1.1,
        "JPY": jpy,
        "CHF": 1.0,
        "CAD": 0.7,
        "AUD": 0.65,
        "NZD": 0.6,
    }


def weekday_market_data(
    start: date,
    end: date,
    *,
    skip: set[date] | None = None,
    usd_prices: dict[date, float] | None = None,
) -> MarketData:
    skipped = skip or set()
    values = usd_prices or {}
    records = {}
    current = start
    while current <= end:
        if current.weekday() < 5 and current not in skipped:
            records[current] = prices(usd=values.get(current, 0.8))
        current += timedelta(days=1)
    return market_data_from_records(records)


def config(
    weights: dict[str, float],
    *,
    start: date = date(2024, 1, 1),
    end: date = date(2024, 1, 26),
) -> BacktestConfig:
    return BacktestConfig(
        initial_capital_eur=10_000,
        transaction_cost_bps=10,
        availability_lag_days=7,
        target_weights=weights,
        start_date=start,
        end_date=end,
    )


def test_eur_only_has_constant_daily_nav_and_no_transactions():
    data = weekday_market_data(date(2024, 1, 1), date(2024, 1, 26))

    result = run_backtest(data, FixedWeightProvider({"EUR": 1.0}), config({"EUR": 1.0}))

    assert len(result.daily_records) == len(data.dates)
    assert result.nav_series.nunique() == 1
    assert result.nav_series.iloc[0] == pytest.approx(10_000)
    assert result.total_fees_eur == 0
    assert all(not execution.rebalance.transactions for execution in result.executions)


def test_backtest_requires_at_least_seven_calendar_days_of_lag():
    with pytest.raises(BacktestConfigError, match="at least 7"):
        BacktestConfig(
            initial_capital_eur=10_000,
            transaction_cost_bps=10,
            availability_lag_days=6,
            target_weights={"EUR": 1.0},
        )


def test_usd_buy_and_hold_has_daily_mark_to_market_and_explicit_fee():
    dates = [date(2024, 1, 15), date(2024, 1, 16)]
    data = market_data_from_records(
        {
            date(2024, 1, 5): prices(usd=0.8),
            date(2024, 1, 12): prices(usd=0.8),
            dates[0]: prices(usd=0.8),
            dates[1]: prices(usd=0.88),
            date(2024, 1, 17): prices(usd=0.88),
        }
    )

    result = run_backtest(data, FixedWeightProvider({"USD": 1.0}), config({"USD": 1.0}))

    first_execution = result.executions[0]
    expected_notional = 10_000 / 1.001
    expected_fee = expected_notional * 0.001
    assert first_execution.execution_date == date(2024, 1, 15)
    assert first_execution.rebalance.transactions[0].notional_eur == pytest.approx(expected_notional)
    assert first_execution.rebalance.operation_fees_eur == pytest.approx(expected_fee)
    nav_15 = result.nav_series.loc[date(2024, 1, 15)]
    nav_16 = result.nav_series.loc[date(2024, 1, 16)]
    assert nav_15 == pytest.approx(10_000 - expected_fee)
    assert nav_16 == pytest.approx(nav_15 * 1.1)
    assert result.total_fees_eur == pytest.approx(expected_fee)
    assert all(record.cash_eur >= 0 for record in result.daily_records)
    assert all(quantity >= 0 for record in result.daily_records for quantity in record.units.values())


def test_weekly_rebalances_execute_after_signals_and_reconcile_fees():
    data = weekday_market_data(date(2024, 1, 1), date(2024, 1, 26))

    class AlternatingProvider:
        def __init__(self):
            self.calls = 0

        def get_target_weights(self, history_until_t, current_state):
            self.calls += 1
            assert history_until_t["market_date"].max() <= date(2024, 1, 19)
            assert not history_until_t["availability_known"].any()
            assert history_until_t["available_at"].isna().all()
            return {"USD": 1.0} if self.calls == 1 else {"EUR": 1.0}

    result = run_backtest(data, AlternatingProvider(), config({"EUR": 1.0}))

    assert [item.signal_date for item in result.executions] == [
        date(2024, 1, 12),
        date(2024, 1, 19),
    ]
    assert [item.execution_date for item in result.executions] == [
        date(2024, 1, 15),
        date(2024, 1, 22),
    ]
    assert [transaction.side for transaction in result.executions[1].rebalance.transactions] == [
        "sell"
    ]
    assert result.total_fees_eur == pytest.approx(
        sum(
            transaction.fee_eur
            for execution in result.executions
            for transaction in execution.rebalance.transactions
        )
    )
    assert all(
        record.nav_eur == pytest.approx(
            record.cash_eur + sum(record.values_eur[currency] for currency in SUPPORTED_CURRENCIES - {"EUR"})
        )
        for record in result.daily_records
    )


def test_usd_to_jpy_conversion_sells_before_buying_with_zero_cash():
    data = weekday_market_data(date(2024, 1, 1), date(2024, 1, 26))

    class USDThenJPYProvider:
        def __init__(self):
            self.calls = 0

        def get_target_weights(self, history_until_t, current_state):
            self.calls += 1
            return {"USD": 1.0} if self.calls == 1 else {"JPY": 1.0}

    result = run_backtest(data, USDThenJPYProvider(), config({"EUR": 1.0}))

    second = result.executions[1].rebalance
    assert result.daily_records[11].cash_eur == pytest.approx(0, abs=1e-8)
    assert [(item.side, item.currency) for item in second.transactions] == [
        ("sell", "USD"),
        ("buy", "JPY"),
    ]
    assert second.operation_fees_eur == pytest.approx(
        sum(item.notional_eur * 0.001 for item in second.transactions)
    )
    assert second.state.cash_eur >= 0
    assert all(quantity >= 0 for quantity in second.state.units.values())


def test_signal_history_uses_only_data_at_least_seven_days_old_and_is_future_invariant():
    start = date(2024, 1, 1)
    end = date(2024, 2, 9)
    baseline = weekday_market_data(start, end)
    changed = weekday_market_data(
        start,
        end,
        usd_prices={
            market_date: 123.0
            for market_date in weekday_market_data(start, end).dates
            if market_date >= date(2024, 1, 20)
        },
    )

    class HistoryProvider:
        def get_target_weights(self, history_until_t, current_state):
            latest_usd = history_until_t.loc[
                history_until_t["currency"] == "USD", "eur_per_unit"
            ].iloc[-1]
            return {"EUR": 1.0} if latest_usd > 10 else {"USD": 1.0}

    run_config = config({"USD": 1.0}, end=end)
    first = run_backtest(baseline, HistoryProvider(), run_config)
    second = run_backtest(changed, HistoryProvider(), run_config)

    prior_decisions = [
        item for item in first.decisions if item.signal_date < date(2024, 1, 20)
    ]
    changed_prior_decisions = [
        item for item in second.decisions if item.signal_date < date(2024, 1, 20)
    ]
    assert prior_decisions == changed_prior_decisions
    assert any(
        left.target_weights != right.target_weights
        for left, right in zip(first.decisions, second.decisions, strict=True)
        if left.information_market_date is not None
        and left.information_market_date >= date(2024, 1, 20)
    )
    for decision in first.decisions + second.decisions:
        if decision.information_market_date is not None:
            assert (decision.signal_date - decision.information_market_date).days >= 7


def test_friday_without_observation_and_market_holiday_execute_on_next_observation():
    data = weekday_market_data(
        date(2024, 1, 1),
        date(2024, 1, 19),
        skip={date(2024, 1, 12), date(2024, 1, 15)},
    )

    result = run_backtest(data, FixedWeightProvider({"USD": 1.0}), config({"USD": 1.0}))

    decision = next(item for item in result.decisions if item.signal_date == date(2024, 1, 12))
    assert decision.status == "executed"
    assert decision.information_market_date == date(2024, 1, 5)
    assert decision.execution_date == date(2024, 1, 16)


def test_last_signal_is_recorded_without_inventing_an_execution():
    data = weekday_market_data(date(2024, 1, 1), date(2024, 1, 26))

    result = run_backtest(data, FixedWeightProvider({"EUR": 1.0}), config({"EUR": 1.0}))

    last = result.decisions[-1]
    assert last.signal_date == date(2024, 1, 26)
    assert last.status == "no_execution_observation"
    assert last.execution_date is None
    assert all(execution.execution_date > execution.signal_date for execution in result.executions)


def test_initial_only_policy_executes_buy_and_hold_once_and_skips_later_fridays():
    data = weekday_market_data(date(2024, 1, 1), date(2024, 1, 26))

    class CountingProvider:
        calls = 0

        def get_target_weights(self, history_until_t, current_state):
            self.calls += 1
            return {"EUR": 0.5, "USD": 0.5}

    provider = CountingProvider()
    run_config = BacktestConfig(
        initial_capital_eur=10_000,
        transaction_cost_bps=10,
        availability_lag_days=7,
        target_weights={"EUR": 1.0},
        start_date=date(2024, 1, 1),
        end_date=date(2024, 1, 26),
        rebalance_policy="initial_only",
    )

    result = run_backtest(data, provider, run_config)

    assert provider.calls == 1
    assert len(result.executions) == 1
    assert result.decisions[0].status == "no_eligible_observation"
    assert result.decisions[1].status == "executed"
    assert all(item.status == "skipped_initial_only" for item in result.decisions[2:])
    assert len(result.executions[0].rebalance.transactions) == 1


def test_provider_current_state_is_reconstructed_at_information_date():
    data = weekday_market_data(date(2024, 1, 1), date(2024, 1, 26))

    class StateCapturingProvider:
        def __init__(self):
            self.states = []

        def get_target_weights(self, history_until_t, current_state):
            self.states.append(current_state)
            return {"USD": 1.0}

    provider = StateCapturingProvider()
    result = run_backtest(data, provider, config({"EUR": 1.0}))

    assert len(result.decisions) - 1 == len(provider.states)
    assert result.decisions[2].information_market_date == date(2024, 1, 12)
    assert all(quantity == 0 for quantity in provider.states[1]["units"].values())
    assert provider.states[1]["cash_eur"] == pytest.approx(10_000)
    assert result.decisions[3].information_market_date == date(2024, 1, 19)
    assert provider.states[2]["units"]["USD"] > 0


def test_runs_are_reproducible_and_report_exports_reconstruct_daily_outputs(tmp_path):
    data = weekday_market_data(date(2024, 1, 1), date(2024, 1, 26))
    provider = FixedWeightProvider({"EUR": 0.5, "USD": 0.5})
    run_config = config({"EUR": 0.5, "USD": 0.5})

    first = run_backtest(data, provider, run_config)
    second = run_backtest(data, FixedWeightProvider({"EUR": 0.5, "USD": 0.5}), run_config)
    output_paths = write_backtest_outputs(first, data, tmp_path)

    pd.testing.assert_series_equal(first.nav_series, second.nav_series)
    assert first.decisions == second.decisions
    assert first.total_fees_eur == second.total_fees_eur
    nav_frame = pd.read_csv(output_paths["nav"])
    allocation_frame = pd.read_csv(output_paths["allocations"])
    assert len(nav_frame) == len(data.dates)
    assert len(allocation_frame) == len(data.dates) * len(SUPPORTED_CURRENCIES)
    assert output_paths["chart"].stat().st_size > 0
    assert output_paths["report"].is_file()
