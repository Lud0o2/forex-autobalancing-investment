"""Validated scenario configuration for reproducible research backtests."""

import math
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import date, datetime
from pathlib import Path
from typing import Any

import yaml

from forex_v1.portfolio.constraints import validate_target_weights


class BacktestConfigError(ValueError):
    """Raised when a backtest scenario is incomplete or invalid."""


@dataclass(frozen=True, slots=True)
class BacktestConfig:
    initial_capital_eur: float
    transaction_cost_bps: float
    availability_lag_days: int
    target_weights: Mapping[str, float]
    start_date: date | None = None
    end_date: date | None = None
    signal_weekday: int = 4
    strategy_name: str = "fixed_weight_test"

    def __post_init__(self) -> None:
        if (
            isinstance(self.initial_capital_eur, bool)
            or not isinstance(self.initial_capital_eur, (int, float))
            or not math.isfinite(float(self.initial_capital_eur))
            or self.initial_capital_eur <= 0
        ):
            raise BacktestConfigError("initial_capital_eur must be finite and positive")
        if (
            isinstance(self.transaction_cost_bps, bool)
            or not isinstance(self.transaction_cost_bps, (int, float))
            or not math.isfinite(float(self.transaction_cost_bps))
            or self.transaction_cost_bps < 0
            or self.transaction_cost_bps >= 10_000
        ):
            raise BacktestConfigError(
                "transaction_cost_bps must be finite and in [0, 10000)"
            )
        if (
            isinstance(self.availability_lag_days, bool)
            or not isinstance(self.availability_lag_days, int)
            or self.availability_lag_days < 7
        ):
            raise BacktestConfigError("availability_lag_days must be an integer of at least 7 days")
        if self.signal_weekday != 4:
            raise BacktestConfigError("The V1 signal weekday must be Friday (4)")
        if self.start_date is not None and (
            isinstance(self.start_date, datetime) or not isinstance(self.start_date, date)
        ):
            raise BacktestConfigError("start_date must be a calendar date or None")
        if self.end_date is not None and (
            isinstance(self.end_date, datetime) or not isinstance(self.end_date, date)
        ):
            raise BacktestConfigError("end_date must be a calendar date or None")
        if self.start_date and self.end_date and self.start_date > self.end_date:
            raise BacktestConfigError("start_date must not be after end_date")
        if not isinstance(self.strategy_name, str) or not self.strategy_name.strip():
            raise BacktestConfigError("strategy_name must be a non-empty string")
        try:
            validated = validate_target_weights(self.target_weights)
        except (TypeError, ValueError) as error:
            raise BacktestConfigError(f"Invalid target_weights: {error}") from error
        object.__setattr__(self, "initial_capital_eur", float(self.initial_capital_eur))
        object.__setattr__(self, "transaction_cost_bps", float(self.transaction_cost_bps))
        object.__setattr__(self, "target_weights", validated)


def load_backtest_config(path: str | Path) -> BacktestConfig:
    """Load a dedicated scenario YAML without changing the project-wide null cost."""
    config_path = Path(path)
    try:
        document = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    except OSError as error:
        raise BacktestConfigError(f"Cannot read backtest config {config_path}: {error}") from error
    except yaml.YAMLError as error:
        raise BacktestConfigError(f"Invalid backtest YAML: {error}") from error
    if not isinstance(document, Mapping):
        raise BacktestConfigError("Backtest config root must be a mapping")
    prices = document.get("portfolio")
    scenario = document.get("scenario")
    if not isinstance(prices, Mapping) or not isinstance(scenario, Mapping):
        raise BacktestConfigError("Backtest config must contain portfolio and scenario mappings")
    fee = scenario.get("transaction_cost_bps")
    if fee is None:
        raise BacktestConfigError("scenario.transaction_cost_bps must be explicitly configured")
    capital = prices.get("initial_capital_eur")
    if isinstance(capital, bool) or not isinstance(capital, (int, float)):
        raise BacktestConfigError("portfolio.initial_capital_eur must be numeric")
    availability_lag = scenario.get("availability_lag_days")
    if isinstance(availability_lag, bool) or not isinstance(availability_lag, int):
        raise BacktestConfigError("scenario.availability_lag_days must be an integer")
    target_weights = scenario.get("target_weights")
    if not isinstance(target_weights, Mapping):
        raise BacktestConfigError("scenario.target_weights must be a mapping")
    return BacktestConfig(
        initial_capital_eur=capital,
        transaction_cost_bps=fee,
        availability_lag_days=availability_lag,
        target_weights=target_weights,
        start_date=_config_date(scenario.get("start_date"), "start_date"),
        end_date=_config_date(scenario.get("end_date"), "end_date"),
        strategy_name=scenario.get("strategy_name", "fixed_weight_test"),
    )


def _config_date(value: Any, field_name: str) -> date | None:
    if value is None:
        return None
    if isinstance(value, datetime) or not isinstance(value, date):
        raise BacktestConfigError(f"{field_name} must be a YAML date or null")
    return value
