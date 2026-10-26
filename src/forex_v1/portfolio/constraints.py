"""Validation for long-only, unleveraged portfolio inputs."""

import math
from collections.abc import Mapping
from typing import Any

from forex_v1.data.schemas import SUPPORTED_CURRENCIES

FOREIGN_CURRENCIES = tuple(sorted(SUPPORTED_CURRENCIES - {"EUR"}))
WEIGHT_TOLERANCE = 1e-9


def _finite_number(value: Any, field_name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{field_name} must be a finite number")
    try:
        result = float(value)
    except OverflowError as error:
        raise ValueError(f"{field_name} must be finite") from error
    if not math.isfinite(result):
        raise ValueError(f"{field_name} must be finite")
    return result


def validate_price_book(prices_eur: Mapping[str, float]) -> dict[str, float]:
    """Validate EUR per unit prices; EUR cash is always worth exactly one EUR."""
    if not isinstance(prices_eur, Mapping):
        raise TypeError("prices_eur must be a mapping")
    if "EUR" not in prices_eur:
        raise ValueError("prices_eur must explicitly contain EUR: 1.0")
    unexpected = set(prices_eur) - SUPPORTED_CURRENCIES
    if unexpected:
        raise ValueError(f"Unsupported price currencies: {sorted(unexpected)}")

    prices: dict[str, float] = {}
    for currency, raw_price in prices_eur.items():
        if not isinstance(currency, str):
            raise TypeError("Price currency codes must be strings")
        price = _finite_number(raw_price, f"price for {currency}")
        if price <= 0:
            raise ValueError(f"price for {currency} must be strictly positive")
        prices[currency] = price
    if prices["EUR"] != 1.0:
        raise ValueError("EUR cash must be valued at exactly 1.0 EUR per EUR")
    return prices


def validate_quantities(units: Mapping[str, float]) -> dict[str, float]:
    """Validate non-negative foreign currency units; EUR is represented by cash."""
    if not isinstance(units, Mapping):
        raise TypeError("units must be a mapping")
    if "EUR" in units:
        raise ValueError("EUR must be represented as cash_eur, not as a currency position")
    unexpected = set(units) - set(FOREIGN_CURRENCIES)
    if unexpected:
        raise ValueError(f"Unsupported position currencies: {sorted(unexpected)}")
    normalized: dict[str, float] = {}
    for currency, raw_quantity in units.items():
        if not isinstance(currency, str):
            raise TypeError("Position currency codes must be strings")
        quantity = _finite_number(raw_quantity, f"quantity for {currency}")
        if quantity < 0:
            raise ValueError(f"quantity for {currency} must not be negative")
        normalized[currency] = quantity
    return normalized


def validate_target_weights(
    target_weights: Mapping[str, float],
) -> dict[str, float]:
    """Validate non-negative target weights summing to one; omitted currencies are zero."""
    if not isinstance(target_weights, Mapping):
        raise TypeError("target_weights must be a mapping")
    if not target_weights:
        raise ValueError("target_weights must not be empty")
    unexpected = set(target_weights) - SUPPORTED_CURRENCIES
    if unexpected:
        raise ValueError(f"Unsupported target currencies: {sorted(unexpected)}")

    weights = {currency: 0.0 for currency in SUPPORTED_CURRENCIES}
    for currency, raw_weight in target_weights.items():
        if not isinstance(currency, str):
            raise TypeError("Target currency codes must be strings")
        weight = _finite_number(raw_weight, f"weight for {currency}")
        if weight < 0 or weight > 1:
            raise ValueError(f"weight for {currency} must be between zero and one")
        weights[currency] = weight
    total = math.fsum(weights.values())
    if not math.isclose(total, 1.0, rel_tol=0.0, abs_tol=WEIGHT_TOLERANCE):
        raise ValueError(f"Target weights must sum to 1.0; got {total:.17g}")
    return weights
