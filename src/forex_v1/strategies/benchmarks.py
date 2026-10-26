"""Deterministic non-ML benchmark providers for Forex V1."""

from __future__ import annotations

import math
from collections.abc import Callable, Mapping

import pandas as pd

from forex_v1.contracts import TargetWeightProvider, TargetWeights
from forex_v1.data.schemas import SUPPORTED_CURRENCIES
from forex_v1.portfolio.constraints import validate_target_weights
from forex_v1.strategies.fixed_weight import FixedWeightProvider

FOREIGN_CURRENCIES = tuple(sorted(SUPPORTED_CURRENCIES - {"EUR"}))
EQUAL_WEIGHT = {currency: 1.0 / 8 for currency in sorted(SUPPORTED_CURRENCIES)}
BENCHMARK_NAMES = (
    "eur_cash",
    "buy_hold_equal_weight",
    "equal_weight_weekly",
    "fixed_4_currencies",
    "momentum_60",
    "inverse_volatility_60",
)
BENCHMARK_REBALANCE_POLICIES = {
    "eur_cash": "weekly",
    "buy_hold_equal_weight": "initial_only",
    "equal_weight_weekly": "weekly",
    "fixed_4_currencies": "weekly",
    "momentum_60": "weekly",
    "inverse_volatility_60": "weekly",
}
REQUIRED_FEATURES = {
    "momentum_60": ("log_return_60",),
    "inverse_volatility_60": ("volatility_60",),
}


def _latest_feature_values(
    history: pd.DataFrame,
    feature_name: str,
) -> dict[str, float | None]:
    values: dict[str, float | None] = {currency: None for currency in FOREIGN_CURRENCIES}
    if history.empty:
        return values
    if "currency" not in history or "market_date" not in history:
        raise ValueError("Strategy history must contain market_date and currency columns")
    for currency in FOREIGN_CURRENCIES:
        column = f"{feature_name}"
        if column not in history:
            continue
        selected = history.loc[
            history["currency"] == currency,
            ["market_date", column],
        ].dropna(subset=[column])
        if selected.empty:
            continue
        raw_value = float(selected.sort_values("market_date")[column].iloc[-1])
        if math.isfinite(raw_value):
            values[currency] = raw_value
    return values


class EurCashProvider:
    """Hold all capital as unremunerated EUR cash."""

    def get_target_weights(
        self,
        history_until_t: pd.DataFrame,
        current_state: Mapping[str, object],
    ) -> TargetWeights:
        return {"EUR": 1.0}


class BuyAndHoldEqualWeightProvider:
    """Target one eighth per currency; use the engine's initial-only policy."""

    def get_target_weights(
        self,
        history_until_t: pd.DataFrame,
        current_state: Mapping[str, object],
    ) -> TargetWeights:
        return dict(EQUAL_WEIGHT)


class EqualWeightWeeklyProvider:
    """Target one eighth per currency on every weekly rebalance."""

    def get_target_weights(
        self,
        history_until_t: pd.DataFrame,
        current_state: Mapping[str, object],
    ) -> TargetWeights:
        return dict(EQUAL_WEIGHT)


class Momentum60Provider:
    """Hold up to three positive 60-observation log returns at 25% each."""

    def get_target_weights(
        self,
        history_until_t: pd.DataFrame,
        current_state: Mapping[str, object],
    ) -> TargetWeights:
        momentum = _latest_feature_values(history_until_t, "log_return_60")
        eligible = sorted(
            (
                (currency, value)
                for currency, value in momentum.items()
                if value is not None and value > 0
            ),
            key=lambda item: (-item[1], item[0]),
        )[:3]
        weights = {currency: 0.0 for currency in SUPPORTED_CURRENCIES}
        for currency, _ in eligible:
            weights[currency] = 0.25
        weights["EUR"] = 1.0 - math.fsum(weights.values())
        return validate_target_weights(weights)


class InverseVolatility60Provider:
    """Allocate by inverse volatility, cap foreign weights, and leave surplus in EUR.

    Missing, non-finite, zero, and negative volatility observations are excluded.
    The remaining inverse-volatility weights are normalized and clipped at the
    configured cap; any clipped surplus stays in EUR rather than being redistributed.
    """

    def __init__(self, *, max_foreign_weight: float = 0.35) -> None:
        if (
            isinstance(max_foreign_weight, bool)
            or not isinstance(max_foreign_weight, (int, float))
            or not math.isfinite(float(max_foreign_weight))
            or not 0 < max_foreign_weight <= 1
        ):
            raise ValueError("max_foreign_weight must be finite and in (0, 1]")
        self.max_foreign_weight = float(max_foreign_weight)

    def get_target_weights(
        self,
        history_until_t: pd.DataFrame,
        current_state: Mapping[str, object],
    ) -> TargetWeights:
        volatility = _latest_feature_values(history_until_t, "volatility_60")
        eligible = {
            currency: value
            for currency, value in volatility.items()
            if value is not None and value > 0
        }
        weights = {currency: 0.0 for currency in SUPPORTED_CURRENCIES}
        if eligible:
            inverse = {currency: 1.0 / value for currency, value in eligible.items()}
            total_inverse = math.fsum(inverse.values())
            for currency, value in inverse.items():
                weights[currency] = min(value / total_inverse, self.max_foreign_weight)
        foreign_total = math.fsum(weights[currency] for currency in FOREIGN_CURRENCIES)
        if foreign_total > 1.0 + 1e-9:
            raise ArithmeticError("Inverse-volatility allocation exceeds total portfolio capital")
        weights["EUR"] = max(0.0, 1.0 - foreign_total)
        return validate_target_weights(weights)


def create_benchmark_provider(
    name: str,
    *,
    max_foreign_weight: float = 0.35,
) -> TargetWeightProvider:
    """Create a reference provider by stable configuration name."""

    normalized = name.strip().lower()
    providers: dict[str, Callable[[], TargetWeightProvider]] = {
        "eur_cash": EurCashProvider,
        "buy_hold_equal_weight": BuyAndHoldEqualWeightProvider,
        "equal_weight_weekly": EqualWeightWeeklyProvider,
        "fixed_4_currencies": FixedWeightProvider,
        "momentum_60": Momentum60Provider,
        "inverse_volatility_60": lambda: InverseVolatility60Provider(
            max_foreign_weight=max_foreign_weight,
        ),
    }
    try:
        return providers[normalized]()
    except KeyError as error:
        raise ValueError(
            f"Unknown benchmark {name!r}; supported benchmarks: {', '.join(BENCHMARK_NAMES)}"
        ) from error


def required_features_for_benchmark(name: str) -> tuple[str, ...]:
    """Return features used to decide whether a benchmark can start."""

    normalized = name.strip().lower()
    if normalized not in BENCHMARK_NAMES:
        raise ValueError(
            f"Unknown benchmark {name!r}; supported benchmarks: {', '.join(BENCHMARK_NAMES)}"
        )
    return REQUIRED_FEATURES.get(normalized, ())
