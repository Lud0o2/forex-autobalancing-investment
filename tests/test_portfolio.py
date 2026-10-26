import math

import pandas as pd
import pytest

from forex_v1.portfolio import (
    PortfolioState,
    create_portfolio,
    fee_rate_from_bps,
    mark_to_market,
    rebalance_from_provider,
    rebalance_portfolio,
    value_portfolio,
)


def price_book(**overrides):
    prices = {"EUR": 1.0, "USD": 0.80, "JPY": 0.006}
    prices.update(overrides)
    return prices


def test_simple_eur_usd_purchase_without_fees():
    state = create_portfolio(10_000, price_book())

    result = rebalance_portfolio(
        state,
        {"EUR": 0.5, "USD": 0.5},
        fee_rate=0.0,
    )

    assert result.state.cash_eur == pytest.approx(5_000)
    assert result.state.units["USD"] == pytest.approx(6_250)
    assert result.nav_after_eur == pytest.approx(10_000)
    assert result.operation_fees_eur == 0
    assert len(result.transactions) == 1
    assert result.transactions[0].side == "buy"
    assert result.transactions[0].notional_eur == pytest.approx(5_000)


def test_mark_to_market_revalues_same_usd_units_and_computes_five_percent_return():
    state = create_portfolio(10_000, price_book())
    invested = rebalance_portfolio(
        state,
        {"EUR": 0.5, "USD": 0.5},
        fee_rate=0.0,
    ).state
    repriced = mark_to_market(invested, price_book(USD=0.88))
    valuation = value_portfolio(repriced)

    assert repriced.units["USD"] == pytest.approx(6_250)
    assert valuation.values_eur["USD"] == pytest.approx(5_500)
    assert repriced.cash_eur == pytest.approx(5_000)
    assert valuation.nav_eur == pytest.approx(10_500)
    assert valuation.nav_eur / value_portfolio(invested).nav_eur - 1 == pytest.approx(0.05)


def test_fee_is_charged_once_on_actual_buy_notional_and_reduces_nav():
    state = create_portfolio(10_000, price_book())
    result = rebalance_portfolio(
        state,
        {"EUR": 0.5, "USD": 0.5},
        fee_rate=fee_rate_from_bps(10),
    )

    assert result.transactions[0].notional_eur == pytest.approx(5_000)
    assert result.transactions[0].fee_eur == pytest.approx(5)
    assert result.operation_fees_eur == pytest.approx(5)
    assert result.state.cumulative_fees_eur == pytest.approx(5)
    assert result.nav_after_eur == pytest.approx(9_995)
    assert result.nav_after_eur == pytest.approx(result.nav_before_eur - result.operation_fees_eur)
    assert result.state.cash_eur == pytest.approx(4_995)
    assert result.valuation.values_eur["USD"] == pytest.approx(5_000)
    assert result.valuation.weights["EUR"] == pytest.approx(4_995 / 9_995)
    assert result.valuation.weights["USD"] == pytest.approx(5_000 / 9_995)


def test_usd_sale_and_jpy_purchase_are_two_eur_routed_conversions():
    initial = create_portfolio(10_000, price_book())
    usd = rebalance_portfolio(
        initial,
        {"EUR": 0.5, "USD": 0.5},
        fee_rate=0,
    ).state

    result = rebalance_portfolio(
        usd,
        {"EUR": 0.5, "JPY": 0.5},
        fee_rate=0.001,
    )

    assert [transaction.side for transaction in result.transactions] == ["sell", "buy"]
    assert [transaction.currency for transaction in result.transactions] == ["USD", "JPY"]
    assert result.transactions[0].notional_eur == pytest.approx(5_000)
    assert result.transactions[0].fee_eur == pytest.approx(5)
    assert result.transactions[1].notional_eur == pytest.approx(5_000)
    assert result.transactions[1].fee_eur == pytest.approx(5)
    assert result.operation_fees_eur == pytest.approx(10)
    assert result.nav_after_eur == pytest.approx(9_990)
    assert result.state.cash_eur == pytest.approx(4_990)
    assert result.state.units["USD"] == 0
    assert result.state.units["JPY"] == pytest.approx(5_000 / 0.006)


def test_exact_target_rebalance_is_noop_and_does_not_require_or_charge_fees():
    initial = create_portfolio(10_000, price_book())
    invested = rebalance_portfolio(
        initial,
        {"EUR": 0.5, "USD": 0.5},
        fee_rate=0,
    ).state

    result = rebalance_portfolio(invested, {"EUR": 0.5, "USD": 0.5})

    assert result.transactions == ()
    assert result.operation_fees_eur == 0
    assert result.state is invested
    assert result.nav_after_eur == pytest.approx(10_000)


def test_fully_foreign_allocation_scales_buys_to_cash_after_fees():
    state = create_portfolio(10_000, price_book())

    result = rebalance_portfolio(state, {"USD": 1.0}, fee_rate=0.001)

    assert result.state.cash_eur == pytest.approx(0, abs=1e-10)
    assert result.transactions[0].notional_eur == pytest.approx(10_000 / 1.001)
    assert result.transactions[0].fee_eur == pytest.approx(10_000 * 0.001 / 1.001)
    assert result.nav_after_eur == pytest.approx(10_000 - result.operation_fees_eur)
    assert all(quantity >= 0 for quantity in result.state.units.values())
    assert math.fsum(result.valuation.weights.values()) == pytest.approx(1)


def test_portfolio_can_remain_all_eur_without_a_configured_cost():
    state = create_portfolio(10_000, price_book())

    result = rebalance_portfolio(state, {"EUR": 1.0})

    assert result.transactions == ()
    assert result.state.cash_eur == 10_000
    assert result.valuation.weights["EUR"] == pytest.approx(1.0)


def test_trade_without_cost_configuration_is_rejected():
    state = create_portfolio(10_000, price_book())

    with pytest.raises(ValueError, match="explicitly configured"):
        rebalance_portfolio(state, {"USD": 1.0})


@pytest.mark.parametrize(
    "weights",
    [
        {"EUR": -0.1, "USD": 1.1},
        {"EUR": 0.4, "USD": 0.4},
        {"EUR": 0.5, "UNKNOWN": 0.5},
        {"EUR": math.inf},
        {"EUR": True},
    ],
)
def test_invalid_allocations_are_rejected(weights):
    state = create_portfolio(10_000, price_book())
    with pytest.raises((TypeError, ValueError)):
        rebalance_portfolio(state, weights, fee_rate=0)


@pytest.mark.parametrize("price", [0, -1, math.inf, -math.inf, math.nan])
def test_invalid_prices_are_rejected(price):
    with pytest.raises((TypeError, ValueError)):
        create_portfolio(10_000, price_book(USD=price))


@pytest.mark.parametrize("capital", [0, -1, math.inf, math.nan, True])
def test_invalid_initial_capital_is_rejected(capital):
    with pytest.raises((TypeError, ValueError)):
        create_portfolio(capital, price_book())


def test_negative_units_and_eur_position_are_rejected():
    with pytest.raises(ValueError, match="not be negative"):
        PortfolioState(
            cash_eur=10_000,
            units={"USD": -1},
            prices_eur=price_book(),
            initial_capital_eur=10_000,
        )
    with pytest.raises(ValueError, match="cash_eur"):
        PortfolioState(
            cash_eur=0,
            units={"EUR": 10_000},
            prices_eur=price_book(),
            initial_capital_eur=10_000,
        )


@pytest.mark.parametrize("fee_rate", [-0.1, 1.0, math.inf, math.nan, True])
def test_invalid_fee_rates_are_rejected(fee_rate):
    state = create_portfolio(10_000, price_book())
    with pytest.raises((TypeError, ValueError)):
        rebalance_portfolio(state, {"USD": 1.0}, fee_rate=fee_rate)


def test_fee_rate_bps_conversion_is_explicit():
    assert fee_rate_from_bps(10) == pytest.approx(0.001)
    assert fee_rate_from_bps(0) == 0
    with pytest.raises(ValueError):
        fee_rate_from_bps(-1)


def test_target_currency_without_price_is_rejected():
    state = create_portfolio(10_000, {"EUR": 1.0})

    with pytest.raises(ValueError, match="Missing EUR price"):
        rebalance_portfolio(state, {"JPY": 1.0}, fee_rate=0)


def test_any_target_weight_provider_can_supply_portfolio_weights():
    class Provider:
        def get_target_weights(self, history_until_t, current_state):
            assert history_until_t.empty
            assert current_state["cash_eur"] == 10_000
            return {"EUR": 0.5, "USD": 0.5}

    state = create_portfolio(10_000, price_book())
    result = rebalance_from_provider(
        state,
        Provider(),
        pd.DataFrame(),
        {"cash_eur": 10_000},
        fee_rate=0,
    )

    assert result.state.units["USD"] == pytest.approx(6_250)
