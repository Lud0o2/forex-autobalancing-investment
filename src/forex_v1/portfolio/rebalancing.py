"""Long-only portfolio rebalancing through EUR cash."""

import math
from collections.abc import Mapping
from dataclasses import dataclass

import pandas as pd

from forex_v1.contracts import TargetWeightProvider, TargetWeights
from forex_v1.portfolio.constraints import (
    FOREIGN_CURRENCIES,
    WEIGHT_TOLERANCE,
    validate_target_weights,
)
from forex_v1.portfolio.costs import validate_fee_rate
from forex_v1.portfolio.positions import PortfolioState
from forex_v1.portfolio.valuation import PortfolioValuation, value_portfolio

TRADE_NOTIONAL_EPSILON = 1e-12


@dataclass(frozen=True, slots=True)
class Transaction:
    """One simulated conversion between EUR and a foreign currency."""

    side: str
    currency: str
    quantity: float
    notional_eur: float
    fee_eur: float


@dataclass(frozen=True, slots=True)
class RebalanceResult:
    """Portfolio state and auditable outcomes from one target rebalance."""

    state: PortfolioState
    transactions: tuple[Transaction, ...]
    nav_before_eur: float
    nav_after_eur: float
    operation_fees_eur: float
    valuation: PortfolioValuation


def rebalance_portfolio(
    state: PortfolioState,
    target_weights: Mapping[str, float],
    *,
    fee_rate: float | None = None,
) -> RebalanceResult:
    """Rebalance to target weights; undersized buy baskets are scaled proportionally."""
    weights = validate_target_weights(target_weights)
    rate = validate_fee_rate(fee_rate) if fee_rate is not None else None
    before = value_portfolio(state)
    desired_units: dict[str, float] = {}
    for currency in FOREIGN_CURRENCIES:
        weight = weights[currency]
        if weight > 0 and currency not in state.prices_eur:
            raise ValueError(f"Missing EUR price for target currency {currency}")
        if currency in state.prices_eur:
            desired_units[currency] = before.nav_eur * weight / state.prices_eur[currency]
        elif state.units[currency] == 0:
            desired_units[currency] = 0.0
        else:
            raise ValueError(f"Missing EUR price for held currency {currency}")

    sells: list[tuple[str, float, float]] = []
    buys: list[tuple[str, float, float]] = []
    for currency in FOREIGN_CURRENCIES:
        current = state.units[currency]
        target = desired_units[currency]
        delta = target - current
        price = state.prices_eur.get(currency)
        if price is None:
            if abs(delta) > 0:
                raise ValueError(f"Missing EUR price for currency {currency}")
            continue
        notional = abs(delta) * price
        if notional <= TRADE_NOTIONAL_EPSILON * max(before.nav_eur, 1.0):
            continue
        if delta < 0:
            sells.append((currency, -delta, notional))
        else:
            buys.append((currency, delta, notional))

    if not sells and not buys:
        return RebalanceResult(
            state=state,
            transactions=(),
            nav_before_eur=before.nav_eur,
            nav_after_eur=before.nav_eur,
            operation_fees_eur=0.0,
            valuation=before,
        )
    if rate is None:
        raise ValueError("A fee_rate must be explicitly configured before any trade")

    cash = state.cash_eur
    units = dict(state.units)
    transactions: list[Transaction] = []
    fees = 0.0
    for currency, quantity, notional in sells:
        fee = notional * rate
        net_proceeds = notional - fee
        cash += net_proceeds
        units[currency] -= quantity
        fees += fee
        transactions.append(Transaction("sell", currency, quantity, notional, fee))

    requested_buy_notional = math.fsum(notional for _, _, notional in buys)
    available_for_buys = max(0.0, cash)
    required_cash = requested_buy_notional * (1.0 + rate)
    buy_scale = min(1.0, available_for_buys / required_cash) if required_cash else 1.0
    for currency, quantity, notional in buys:
        actual_notional = notional * buy_scale
        actual_quantity = quantity * buy_scale
        fee = actual_notional * rate
        cash -= actual_notional + fee
        units[currency] += actual_quantity
        fees += fee
        transactions.append(
            Transaction("buy", currency, actual_quantity, actual_notional, fee)
        )

    cash_tolerance = max(1e-10, before.nav_eur * 1e-12)
    if cash < -cash_tolerance:
        raise ArithmeticError("Rebalancing would spend more EUR cash than available")
    if cash < 0:
        cash = 0.0
    final_state = PortfolioState(
        cash_eur=cash,
        units=units,
        prices_eur=state.prices_eur,
        initial_capital_eur=state.initial_capital_eur,
        cumulative_fees_eur=state.cumulative_fees_eur + fees,
    )
    after = value_portfolio(final_state)
    expected_nav = before.nav_eur - fees
    if not math.isclose(after.nav_eur, expected_nav, rel_tol=1e-12, abs_tol=cash_tolerance):
        raise ArithmeticError(
            f"NAV conservation failed: {after.nav_eur} != {before.nav_eur} - {fees}"
        )
    if after.nav_eur > before.nav_eur + cash_tolerance:
        raise ArithmeticError("Rebalancing created portfolio value")
    if any(quantity < 0 for quantity in final_state.units.values()):
        raise ArithmeticError("Rebalancing created a short currency position")
    if not math.isclose(math.fsum(transaction.fee_eur for transaction in transactions), fees):
        raise ArithmeticError("Transaction fee totals are inconsistent")
    if not math.isclose(
        math.fsum(after.weights.values()),
        1.0,
        rel_tol=0.0,
        abs_tol=WEIGHT_TOLERANCE,
    ):
        raise ArithmeticError("Realized portfolio weights do not sum to one")
    return RebalanceResult(
        state=final_state,
        transactions=tuple(transactions),
        nav_before_eur=before.nav_eur,
        nav_after_eur=after.nav_eur,
        operation_fees_eur=fees,
        valuation=after,
    )


def rebalance_from_provider(
    state: PortfolioState,
    provider: TargetWeightProvider,
    history_until_t: pd.DataFrame,
    current_state: Mapping[str, object],
    *,
    fee_rate: float | None = None,
) -> RebalanceResult:
    """Ask any structural TargetWeightProvider for weights, then rebalance them."""
    target_weights: TargetWeights = provider.get_target_weights(history_until_t, current_state)
    return rebalance_portfolio(state, target_weights, fee_rate=fee_rate)
