from datetime import UTC, date, datetime

import pytest

from forex_v1.data.schemas import FXObservation, PriceType
from forex_v1.data.validation import validate_observations


def make_observation(**overrides):
    values = {
        "market_date": date(2025, 1, 2),
        "base_currency": "EUR",
        "quote_currency": "USD",
        "quote_per_base": 1.08,
        "source": "test",
        "price_type": PriceType.REFERENCE,
    }
    values.update(overrides)
    return FXObservation(**values)


def test_valid_observation_is_accepted():
    observation = make_observation()

    assert observation.quote_per_base == 1.08
    assert observation.base_currency == "EUR"
    assert observation.quote_currency == "USD"


@pytest.mark.parametrize("rate", [0, -1, float("nan"), float("inf"), float("-inf")])
def test_invalid_rates_are_rejected(rate):
    with pytest.raises(ValueError, match="finite and strictly positive"):
        make_observation(quote_per_base=rate)


def test_unsupported_currency_is_rejected():
    with pytest.raises(ValueError, match="supported ISO currency codes"):
        make_observation(base_currency="XYZ")


def test_duplicate_source_pair_market_date_and_price_type_are_rejected():
    observation = make_observation()

    with pytest.raises(ValueError, match="Duplicate observation"):
        validate_observations([observation, observation])


def test_inverse_pair_is_a_distinct_observation():
    eur_usd = make_observation()
    usd_eur = make_observation(
        base_currency="USD",
        quote_currency="EUR",
        quote_per_base=1 / 1.08,
    )

    validate_observations([eur_usd, usd_eur])


def test_naive_availability_timestamp_is_rejected():
    with pytest.raises(ValueError, match="explicit timezone"):
        make_observation(
            available_at=datetime(2025, 1, 2, 15, 0),
            availability_known=True,
        )


def test_market_date_and_availability_timestamp_remain_distinct():
    available_at = datetime(2025, 1, 3, 8, 30, tzinfo=UTC)
    observation = make_observation(
        market_date=date(2025, 1, 2),
        available_at=available_at,
        availability_known=True,
    )

    assert observation.market_date == date(2025, 1, 2)
    assert observation.available_at == available_at
    assert observation.market_date != observation.available_at.date()


def test_unknown_availability_is_explicit():
    observation = make_observation()

    assert observation.availability_known is False
    assert observation.available_at is None


def test_market_date_must_not_be_a_timestamp():
    with pytest.raises(TypeError, match="calendar date"):
        make_observation(market_date=datetime(2025, 1, 2, tzinfo=UTC))
