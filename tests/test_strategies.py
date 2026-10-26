from datetime import date

import numpy as np
import pandas as pd
import pytest

from forex_v1.data.schemas import SUPPORTED_CURRENCIES
from forex_v1.strategies import (
    BENCHMARK_REBALANCE_POLICIES,
    BuyAndHoldEqualWeightProvider,
    EqualWeightWeeklyProvider,
    EurCashProvider,
    InverseVolatility60Provider,
    Momentum60Provider,
    create_benchmark_provider,
)

FOREIGN_CURRENCIES = sorted(SUPPORTED_CURRENCIES - {"EUR"})


def history(values: dict[str, float | None], feature: str) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "market_date": date(2024, 1, 2),
                "currency": currency,
                "eur_per_unit": 1.0,
                feature: value,
            }
            for currency, value in values.items()
        ]
    )


def assert_valid_long_only(weights: dict[str, float]) -> None:
    assert set(weights) <= SUPPORTED_CURRENCIES
    assert all(0 <= weight <= 1 for weight in weights.values())
    assert sum(weights.values()) == pytest.approx(1)


def test_static_benchmarks_return_the_expected_configurable_targets():
    empty = pd.DataFrame()
    state: dict[str, object] = {}
    eur = EurCashProvider().get_target_weights(empty, state)
    hold = BuyAndHoldEqualWeightProvider().get_target_weights(empty, state)
    weekly = EqualWeightWeeklyProvider().get_target_weights(empty, state)

    assert eur == {"EUR": 1.0}
    assert set(hold) == SUPPORTED_CURRENCIES
    assert all(weight == pytest.approx(0.125) for weight in hold.values())
    assert weekly == hold
    for weights in (eur, hold, weekly):
        assert_valid_long_only(dict(weights))


def test_momentum_selects_top_three_positive_log_returns_and_keeps_other_weight_in_eur():
    values = {
        "USD": 0.5,
        "GBP": 0.4,
        "JPY": 0.3,
        "CHF": 0.2,
        "CAD": -0.1,
        "AUD": 0.0,
        "NZD": None,
    }

    weights = Momentum60Provider().get_target_weights(history(values, "log_return_60"), {})

    assert weights["USD"] == pytest.approx(0.25)
    assert weights["GBP"] == pytest.approx(0.25)
    assert weights["JPY"] == pytest.approx(0.25)
    assert weights["EUR"] == pytest.approx(0.25)
    assert_valid_long_only(dict(weights))


def test_momentum_keeps_unused_slots_in_eur_and_ties_are_deterministic():
    values = {currency: None for currency in FOREIGN_CURRENCIES}
    values.update({"USD": 0.1, "GBP": 0.1})

    weights = Momentum60Provider().get_target_weights(history(values, "log_return_60"), {})

    assert weights["GBP"] == pytest.approx(0.25)
    assert weights["USD"] == pytest.approx(0.25)
    assert weights["EUR"] == pytest.approx(0.5)
    assert sum(weight > 0 for currency, weight in weights.items() if currency != "EUR") == 2


def test_inverse_volatility_prefers_low_volatility_caps_exposure_and_keeps_surplus_in_eur():
    values = {currency: 1.0 for currency in FOREIGN_CURRENCIES}
    values.update({"USD": 0.01, "JPY": 0.0, "CHF": None, "CAD": np.nan})

    weights = InverseVolatility60Provider(max_foreign_weight=0.35).get_target_weights(
        history(values, "volatility_60"),
        {},
    )

    assert weights["USD"] == pytest.approx(0.35)
    assert weights["JPY"] == 0
    assert weights["CHF"] == 0
    assert weights["CAD"] == 0
    assert weights["EUR"] > 0
    assert all(
        weights[currency] <= 0.35
        for currency in FOREIGN_CURRENCIES
    )
    assert_valid_long_only(dict(weights))


def test_inverse_volatility_with_only_missing_or_zero_values_holds_eur():
    values = {currency: 0.0 for currency in FOREIGN_CURRENCIES}
    values["USD"] = None

    weights = InverseVolatility60Provider().get_target_weights(
        history(values, "volatility_60"),
        {},
    )

    assert weights == {**{currency: 0.0 for currency in FOREIGN_CURRENCIES}, "EUR": 1.0}


def test_inverse_volatility_rounding_never_creates_a_negative_eur_allocation():
    rng = np.random.default_rng(702)
    provider = InverseVolatility60Provider()
    for _ in range(100):
        values = {
            currency: float(10 ** rng.uniform(-4, 0))
            for currency in FOREIGN_CURRENCIES
        }
        weights = provider.get_target_weights(history(values, "volatility_60"), {})
        assert weights["EUR"] >= 0
        assert all(weights[currency] <= 0.35 for currency in FOREIGN_CURRENCIES)
        assert_valid_long_only(dict(weights))


def test_benchmark_factory_supports_configuration_names_and_buy_hold_policy():
    fixed_weights = create_benchmark_provider("fixed_4_currencies").get_target_weights(
        pd.DataFrame(),
        {},
    )
    assert {currency: fixed_weights[currency] for currency in ("EUR", "USD", "GBP", "JPY")} == (
        pytest.approx({"EUR": 0.25, "USD": 0.25, "GBP": 0.25, "JPY": 0.25})
    )
    assert_valid_long_only(dict(fixed_weights))
    assert BENCHMARK_REBALANCE_POLICIES["buy_hold_equal_weight"] == "initial_only"
    assert BENCHMARK_REBALANCE_POLICIES["equal_weight_weekly"] == "weekly"


def test_inverse_volatility_rejects_invalid_concentration_caps():
    for cap in (0.0, -0.1, 1.1, np.nan):
        with pytest.raises(ValueError, match="max_foreign_weight"):
            InverseVolatility60Provider(max_foreign_weight=cap)
