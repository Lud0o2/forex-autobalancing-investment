"""Immutable state representation for EUR-cash and foreign currency holdings."""

import math
from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType

from forex_v1.portfolio.constraints import (
    FOREIGN_CURRENCIES,
    validate_price_book,
    validate_quantities,
)


@dataclass(frozen=True, slots=True)
class PortfolioState:
    """Current cash, foreign units, last known EUR prices, and fee history."""

    cash_eur: float
    units: Mapping[str, float]
    prices_eur: Mapping[str, float]
    initial_capital_eur: float
    cumulative_fees_eur: float = 0.0

    def __post_init__(self) -> None:
        numeric_fields = {
            "cash_eur": self.cash_eur,
            "initial_capital_eur": self.initial_capital_eur,
            "cumulative_fees_eur": self.cumulative_fees_eur,
        }
        normalized: dict[str, float] = {}
        for name, value in numeric_fields.items():
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise TypeError(f"{name} must be a finite number")
            try:
                numeric = float(value)
            except OverflowError as error:
                raise ValueError(f"{name} must be finite") from error
            if not math.isfinite(numeric):
                raise ValueError(f"{name} must be finite")
            normalized[name] = numeric
        if normalized["cash_eur"] < 0:
            raise ValueError("cash_eur must not be negative")
        if normalized["initial_capital_eur"] <= 0:
            raise ValueError("initial_capital_eur must be strictly positive")
        if normalized["cumulative_fees_eur"] < 0:
            raise ValueError("cumulative_fees_eur must not be negative")

        prices = validate_price_book(self.prices_eur)
        positions = validate_quantities(self.units)
        missing_held_prices = {
            currency
            for currency, quantity in positions.items()
            if quantity > 0 and currency not in prices
        }
        if missing_held_prices:
            raise ValueError(f"Missing prices for held currencies: {sorted(missing_held_prices)}")
        complete_units = {currency: positions.get(currency, 0.0) for currency in FOREIGN_CURRENCIES}
        object.__setattr__(self, "cash_eur", normalized["cash_eur"])
        object.__setattr__(self, "initial_capital_eur", normalized["initial_capital_eur"])
        object.__setattr__(self, "cumulative_fees_eur", normalized["cumulative_fees_eur"])
        object.__setattr__(self, "units", MappingProxyType(complete_units))
        object.__setattr__(self, "prices_eur", MappingProxyType(prices))


def create_portfolio(
    initial_capital_eur: float,
    prices_eur: Mapping[str, float],
) -> PortfolioState:
    """Start with the entire capital as EUR cash and no foreign borrowing."""
    return PortfolioState(
        cash_eur=initial_capital_eur,
        units={},
        prices_eur=prices_eur,
        initial_capital_eur=initial_capital_eur,
    )
