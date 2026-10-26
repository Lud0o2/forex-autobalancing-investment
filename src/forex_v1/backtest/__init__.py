"""Historical simulation interfaces."""

from forex_v1.backtest.config import BacktestConfig, BacktestConfigError, load_backtest_config
from forex_v1.backtest.data import (
    BacktestDataError,
    MarketData,
    load_market_data,
    market_data_from_records,
)
from forex_v1.backtest.engine import (
    BacktestDailyRecord,
    BacktestDecision,
    BacktestExecution,
    BacktestResult,
    build_history_frame,
    generate_signal_dates,
    run_backtest,
)
from forex_v1.backtest.strategy import FixedWeightProvider

__all__ = [
    "BacktestConfig",
    "BacktestConfigError",
    "BacktestDailyRecord",
    "BacktestDataError",
    "BacktestDecision",
    "BacktestExecution",
    "BacktestResult",
    "FixedWeightProvider",
    "MarketData",
    "build_history_frame",
    "generate_signal_dates",
    "load_backtest_config",
    "load_market_data",
    "market_data_from_records",
    "run_backtest",
]
