"""Portfolio valuation and actual-weight calculations."""

import math
from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType

from forex_v1.data.schemas import SUPPORTED_CURRENCIES
from forex_v1.portfolio.constraints import validate_price_book
from forex_v1.portfolio.positions import PortfolioState


@dataclass(frozen=True, slots=True)
class PortfolioValuation:
    """EUR value and realized weight by currency at the state's current prices."""

    nav_eur: float
    values_eur: Mapping[str, float]
    weights: Mapping[str, float]


def value_portfolio(state: PortfolioState) -> PortfolioValuation:
    """Mark cash and foreign units to EUR using only the state's supplied prices."""
    values = {"EUR": state.cash_eur}
    for currency in sorted(SUPPORTED_CURRENCIES - {"EUR"}):
        quantity = state.units[currency]
        if quantity == 0:
            values[currency] = 0.0
            continue
        try:
            price = state.prices_eur[currency]
        except KeyError as error:
            raise ValueError(f"Missing EUR price for held currency {currency}") from error
        values[currency] = quantity * price
    nav = math.fsum(values.values())
    if not math.isfinite(nav) or nav < 0:
        raise ValueError("Portfolio NAV must be finite and non-negative")
    weights = (
        {currency: value / nav for currency, value in values.items()}
        if nav > 0
        else {currency: 0.0 for currency in values}
    )
    return PortfolioValuation(
        nav_eur=nav,
        values_eur=MappingProxyType(values),
        weights=MappingProxyType(weights),
    )


def mark_to_market(
    state: PortfolioState,
    prices_eur: Mapping[str, float],
) -> PortfolioState:
    """Return the same holdings marked at explicitly supplied new EUR prices."""
    prices = validate_price_book(prices_eur)
    return PortfolioState(
        cash_eur=state.cash_eur,
        units=state.units,
        prices_eur=prices,
        initial_capital_eur=state.initial_capital_eur,
        cumulative_fees_eur=state.cumulative_fees_eur,
    )
