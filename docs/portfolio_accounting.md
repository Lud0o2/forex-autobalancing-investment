# Multi-currency portfolio accounting (step 6A)

## Scope

The portfolio package handles explicit spot holdings in EUR, USD, GBP, JPY, CHF,
CAD, AUD, and NZD. It is independent of strategies and models; it consumes target
weights from any object implementing `TargetWeightProvider`, or directly from a
mapping. It does not choose allocations, schedule historical decisions, simulate
weekly rebalances, generate broker orders, or implement step 6B.

The caller supplies a current EUR-per-unit price for EUR and every foreign currency
that is held or traded. EUR must be exactly `1.0 EUR per EUR`. Prices must be finite
and strictly positive. Feature indicators are never used as execution prices.
FRED-derived rates are indicative reference prices, not proof of executable broker
quotes.

## State and valuation

`PortfolioState` records:

- EUR cash, held separately from foreign currency units.
- Non-negative units for each foreign currency (zero units are filled into the
  state representation).
- The supplied EUR price book.
- Initial capital and cumulative conversion fees.

EUR cannot be stored as a foreign position. For each foreign currency `i`:

```text
value_eur[i] = units[i] * eur_per_unit[i]
NAV_EUR = cash_eur + sum(value_eur[i])
weight[i] = value_eur[i] / NAV_EUR
weight[EUR] = cash_eur / NAV_EUR
```

Weights are derived from realized holdings and are not assumed to equal target
weights after fees. A state cannot contain negative cash or units. The engine
applies no leverage, borrowing, short selling, interest, or carry.

`mark_to_market` replaces prices while preserving cash and units; it does not
rebalance or create a transaction. Thus an FX valuation change affects NAV without
orders or costs.

## Target allocations and constraints

Target weights must be finite, between 0 and 1 inclusive, and sum to 1 within
`1e-9` absolute tolerance. An omitted currency has a zero target weight. Unknown
currencies and malformed allocations are rejected. No concentration limit is
invented. The caller supplies the provider's context/history and the provider
returns only currency weights; the portfolio validates and executes that mapping.

## Fees and conversions

The cost model is proportional to the actual EUR notional traded:

```text
fee_eur = traded_notional_eur * fee_rate
```

`fee_rate` is a fraction, not a percentage or basis-point count. For example,
10 bps is `0.001`, converted with `fee_rate_from_bps(10)`. An operation that
generates a trade must supply a fee rate explicitly. There is no implicit free
trading. An explicit `0.0` is supported for mathematical tests. Rates must be
finite and in `[0, 1)`; larger rates cannot represent a sale that can settle from
its proceeds.

The existing `configs/v1.yaml` has `transaction_cost_bps: null`. That means the
configuration currently makes no cost assumption; do not run a trade until a
caller explicitly supplies a cost. This engine does not silently substitute the
plan's indicative default or any broker fee.

Conversions are modeled conceptually as currency-to-EUR sales and EUR-to-currency
purchases. A USD-to-JPY change therefore consists of a USD sale and a JPY buy;
the fee applies separately to each actual notional. This EUR routing is a
simplification and does not necessarily represent the best real broker routing,
spread, execution price, or settlement mechanics.

## Rebalancing

The engine calculates desired foreign notionals from pre-operation NAV:

```text
desired_value_eur[i] = NAV_before * target_weight[i]
desired_units[i] = desired_value_eur[i] / eur_per_unit[i]
```

It sells reductions first. Sale proceeds are added to cash after the sale fee.
It then calculates required purchases. If the complete buy basket and its fees
exceed available cash, all requested buys are reduced by the same deterministic
pro-rata scale:

```text
buy_scale = min(1, cash_available / (sum(buy_notionals) * (1 + fee_rate)))
```

This preserves the relative sizes of requested purchases and never borrows. Any
unspent cash remains in EUR; realized weights can differ from targets because of
fees and buy scaling. Transactions report side, currency, units, EUR notional, and
fee. Fees are accumulated independently from currency valuation changes.

An allocation that already matches realized weights within a small notional
tolerance creates no transactions and incurs no fee; the fee rate can remain
unconfigured for that no-op. When trades are needed and no fee rate is supplied,
the operation fails explicitly.

## Accounting invariants

After each rebalance the engine verifies:

- Cash and each foreign unit balance are non-negative.
- NAV is finite and non-negative.
- Realized currency weights sum to one when NAV is positive.
- Transaction fee totals reconcile with cumulative fees.
- At unchanged prices, `NAV_after = NAV_before - operation_fees`.
- Rebalancing cannot increase NAV or create capital.

These checks use floating-point tolerances; they are accounting guards, not a
broker settlement ledger.

## Boundaries and limitations

- No historical weekly simulation, signal/execution chronology, performance
  metrics, strategies, or model code is included in step 6A.
- The caller must supply prices explicitly. Missing prices for an actual holding or
  positive target allocation are errors; there is no price fill or fallback.
- The model charges proportional conversion costs only. It does not include
  spreads, slippage, minimum ticket fees, taxes, interest, or carry.
- FRED H.10 rates are indicative, and historical availability is not certified
  point-in-time. A future backtest must address this separately.
- This is a research accounting engine, not a trading or broker execution system.
