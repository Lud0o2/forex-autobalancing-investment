"""Causal, dimensionless technical indicators for one currency price history."""

from collections.abc import Mapping
from typing import Any

import numpy as np
import pandas as pd

SUPPORTED_INDICATORS = frozenset(
    {
        "log_return",
        "rsi",
        "momentum",
        "sma_relative",
        "volatility",
        "bollinger_zscore",
        "channel_position",
        "drawdown",
    }
)


def _validate_price_series(prices: pd.Series) -> pd.Series:
    if not isinstance(prices, pd.Series):
        raise TypeError("prices must be a pandas Series")
    if prices.empty:
        raise ValueError("prices must not be empty")
    if prices.index.has_duplicates or not prices.index.is_monotonic_increasing:
        raise ValueError("price index must be unique and sorted chronologically")
    numeric = pd.to_numeric(prices, errors="coerce").astype(float)
    if numeric.isna().any() or not np.isfinite(numeric.to_numpy()).all():
        raise ValueError("prices must contain only finite, non-missing values")
    if (numeric <= 0).any():
        raise ValueError("prices must be strictly positive")
    return numeric


def _window_name(indicator: str, window: int) -> str:
    return f"{indicator}_{window}"


def calculate_rsi(prices: pd.Series, window: int = 14) -> pd.Series:
    """Return RSI scaled to [0, 1]; flat windows use 0.5, gain/loss-only use 1/0."""
    prices = _validate_price_series(prices)
    if isinstance(window, bool) or not isinstance(window, int) or window < 1:
        raise ValueError("window must be a positive integer")
    changes = prices.diff()
    gains = changes.clip(lower=0)
    losses = -changes.clip(upper=0)
    average_gain = gains.rolling(window=window, min_periods=window).mean()
    average_loss = losses.rolling(window=window, min_periods=window).mean()
    denominator = average_gain + average_loss
    rsi = average_gain.div(denominator)
    flat = denominator.eq(0) & denominator.notna()
    rsi = rsi.mask(flat, 0.5)
    return rsi.rename(_window_name("rsi", window))


def calculate_log_return(prices: pd.Series, window: int = 1) -> pd.Series:
    """Log price change over the requested number of observations."""
    prices = _validate_price_series(prices)
    if isinstance(window, bool) or not isinstance(window, int) or window < 1:
        raise ValueError("window must be a positive integer")
    return pd.Series(
        np.log(prices / prices.shift(window)),
        index=prices.index,
    ).rename(_window_name("log_return", window))


def calculate_momentum(prices: pd.Series, window: int = 20) -> pd.Series:
    """Cumulative log return over a number of observations (same unit-free form as log return)."""
    prices = _validate_price_series(prices)
    if isinstance(window, bool) or not isinstance(window, int) or window < 1:
        raise ValueError("window must be a positive integer")
    return pd.Series(
        np.log(prices / prices.shift(window)),
        index=prices.index,
    ).rename(_window_name("momentum", window))


def calculate_sma_relative(prices: pd.Series, window: int = 20) -> pd.Series:
    """Return price / trailing simple moving average - 1."""
    prices = _validate_price_series(prices)
    if isinstance(window, bool) or not isinstance(window, int) or window < 1:
        raise ValueError("window must be a positive integer")
    average = prices.rolling(window=window, min_periods=window).mean()
    return (prices / average - 1.0).rename(_window_name("sma_relative", window))


def calculate_volatility(
    prices: pd.Series,
    window: int = 20,
    *,
    annualization_observations: int = 252,
) -> pd.Series:
    """Sample std of rolling one-observation log returns times sqrt(annualization factor)."""
    prices = _validate_price_series(prices)
    if isinstance(window, bool) or not isinstance(window, int) or window < 2:
        raise ValueError("volatility window must be an integer of at least 2")
    if (
        isinstance(annualization_observations, bool)
        or not isinstance(annualization_observations, int)
        or annualization_observations < 1
    ):
        raise ValueError("annualization_observations must be a positive integer")
    log_returns = pd.Series(
        np.log(prices / prices.shift(1)),
        index=prices.index,
    )
    volatility = (
        log_returns.rolling(window=window, min_periods=window).std(ddof=1)
        * np.sqrt(annualization_observations)
    )
    return volatility.rename(_window_name("volatility", window))


def calculate_bollinger_zscore(prices: pd.Series, window: int = 20) -> pd.Series:
    """Distance from trailing mean in population standard deviations; flat window uses 0."""
    prices = _validate_price_series(prices)
    if isinstance(window, bool) or not isinstance(window, int) or window < 2:
        raise ValueError("window must be an integer of at least 2")
    rolling = prices.rolling(window=window, min_periods=window)
    mean = rolling.mean()
    deviation = rolling.std(ddof=0)
    zscore = (prices - mean).div(deviation.where(deviation.ne(0)))
    return zscore.mask(deviation.eq(0) & deviation.notna(), 0.0).rename(
        _window_name("bollinger_zscore", window)
    )


def calculate_channel_position(prices: pd.Series, window: int = 20) -> pd.Series:
    """Position in trailing low/high channel; zero-range windows use 0.5."""
    prices = _validate_price_series(prices)
    if isinstance(window, bool) or not isinstance(window, int) or window < 1:
        raise ValueError("window must be a positive integer")
    rolling = prices.rolling(window=window, min_periods=window)
    low = rolling.min()
    high = rolling.max()
    width = high - low
    position = (prices - low).div(width.where(width.ne(0)))
    return position.mask(width.eq(0) & width.notna(), 0.5).rename(
        _window_name("channel_position", window)
    )


def calculate_drawdown(prices: pd.Series, window: int = 60) -> pd.Series:
    """Return price / trailing rolling maximum - 1."""
    prices = _validate_price_series(prices)
    if isinstance(window, bool) or not isinstance(window, int) or window < 1:
        raise ValueError("window must be a positive integer")
    rolling_maximum = prices.rolling(window=window, min_periods=window).max()
    return (prices / rolling_maximum - 1.0).rename(_window_name("drawdown", window))


def calculate_technical_features(
    prices: pd.Series,
    configuration: Mapping[str, Any],
) -> pd.DataFrame:
    """Calculate only enabled configured indicators, preserving the price index."""
    prices = _validate_price_series(prices)
    annualization = configuration["annualization_observations"]
    output: dict[str, pd.Series] = {}
    functions = {
        "log_return": calculate_log_return,
        "rsi": calculate_rsi,
        "momentum": calculate_momentum,
        "sma_relative": calculate_sma_relative,
        "volatility": calculate_volatility,
        "bollinger_zscore": calculate_bollinger_zscore,
        "channel_position": calculate_channel_position,
        "drawdown": calculate_drawdown,
    }
    for indicator, settings in configuration["indicators"].items():
        if not settings["enabled"]:
            continue
        for window in settings["windows"]:
            if indicator == "volatility":
                values = functions[indicator](
                    prices,
                    window,
                    annualization_observations=annualization,
                )
            else:
                values = functions[indicator](prices, window)
            output[values.name] = values
    return pd.DataFrame(output, index=prices.index)
