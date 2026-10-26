# MLP V1 — allocation directe (étape 8B)

## Scope and reproducibility

The first PyTorch MLP maps the existing 91 normalized 8A features directly to
eight long-only currency weights. It is trained separately for each of the
three development-only walk-forward folds:

```powershell
.\.venv\Scripts\python.exe -m forex_v1.models.mlp_training
```

The command reads `configs/mlp_v1.yaml`, the immutable 8A dataset/fold CSVs and
scalers, and local EUR/feature histories. It refuses a dataset without a
development/test boundary, mismatched source hashes, missing scaler provenance,
unpurged targets, or validation labels extending past the development end. The
final test period beginning 2019-10-25 is not passed to the trainer or the
validation backtests. No automatic tuning, neural architecture search, or live
trading is performed.

Each fold starts from a new seeded model, trains in chronological row order with
full-sequence gradients, and uses only that fold's validation loss for early
stopping. There is no minibatch shuffling. The checkpoint contains only a
PyTorch `state_dict`; its JSON manifest stores the architecture, exact feature
order, scaler values and train-only fit declaration, seed, training history,
validation metrics, date ranges, purge count, and hashes of the input data.
Checkpoint paths are content-addressed and immutable.

## Architecture

The default network is `91 → 32 → 16 → 8`, with ReLU after each hidden affine
layer and Softmax at the output. Output order is fixed as EUR, USD, GBP, JPY,
CHF, CAD, AUD, NZD. Consequently output weights are non-negative and sum to one,
with no short positions or leverage. Hidden layer widths are configurable;
there is no LSTM, Transformer, or additional allocation model in this step.

## Differentiable financial objective

For chronological example `t`, the gross portfolio return is
`g_t = sum_i(w_t[i] * Y_t[i])`. Let `q_t` be the pre-trade allocation: EUR-only
for the first example, and thereafter the previous target weights drifted by the
previous example's asset returns:

`q_(t+1)[i] = w_t[i] * (1 + Y_t[i]) / sum_j(w_t[j] * (1 + Y_t[j]))`.

The differentiable foreign-currency turnover proxy is
`u_t = sum_(i != EUR)(abs(w_t[i] - q_t[i]))`. With configured fee rate
`c = transaction_cost_bps / 10_000`, the approximate net growth factor is
`F_t = (1 - c * u_t) * (1 + g_t)` and the default loss is
`-mean(log(F_t))`. Thus an initial purchase from 100% EUR incurs a cost, and
costs are applied once rather than subtracted a second time from returns.

This proxy is differentiable with respect to the MLP weights and keeps the
allocation drift in the gradient path. It is **not** the accounting engine:
actual fee notionals, cash constraints, and post-fee quantities can differ.
Tests compare the approximation with 6A on a simple transaction and quantify
the small discrepancy. Reported validation performance always comes from
the exact existing portfolio/backtest engine, never from this approximate loss.

An optional drawdown penalty can be configured, but defaults to zero. A penalty
or a low training drawdown does not guarantee that a validation drawdown remains
below 20%.

## Exact fold validation

For each fold the model is adapted to the shared `TargetWeightProvider` contract.
The backtest uses the same Friday signals and seven-calendar-day information lag,
matching the 8A input and execution dates. The valuation period includes a
pre-signal 10,000 EUR NAV observation and the last label endpoint. The decision
calendar ends at the final validation signal, so the final execution endpoint is
marked without creating an extra out-of-fold decision. This uses a separate
`signal_end_date`; absent this boundary the generic backtester would otherwise
consider an additional Friday signal while valuing through the next execution.

EUR Cash, Equal Weight Weekly, and genuine initial-only Buy & Hold are each
rerun on exactly the same market dates, capital, lag, and configured 10 bps
fees. Buy & Hold intentionally trades once; its NAV date series is still common
with the other strategies. CAGR, total return, maximum drawdown, Sharpe,
annualized volatility, fees, gross turnover, and final NAV in the report are
calculated from each strategy's exact daily NAV and transaction journal. Fold
summary values are unweighted arithmetic means across the three independent
validation windows; they are not a continuous stitched portfolio.

The report also records training/validation loss histories, allocation averages
and concentration, mean number of foreign currencies held, weekly allocation
changes, exact turnover and fees, input sensitivity, and diagnostic flags for
cash-only, single-currency, near-static, or highly changing behavior. These
diagnostics describe the model and do not trigger post-hoc parameter tuning.

## Limitations

FRED-derived rates are indicative reference values, not certified executable
prices, and actual historical publication availability is unknown. Seven days
is the conservative research lag inherited from the project, not a point-in-time
availability guarantee. Spread, slippage, interest, carry, and taxes are not
modeled. The short initial development folds, differentiable fee approximation,
and chosen objective can overfit; training loss improvement is not evidence of
future profitability. The final 25% is reserved and was not evaluated.
