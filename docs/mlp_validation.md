# MLP V1 validation protocol (step 8C)

## Nested chronological validation

The step 8B training procedure used each outer walk-forward validation fold to
select an epoch checkpoint and then reported performance on that same fold. The
reported outer-fold financial results were therefore involved in model
selection and are optimistic estimates, not untouched out-of-sample results.
The outer folds had already been reviewed by people and remain part of
development.

Step 8C preserves the step 8B architecture, optimizer, learning rate, random
seed, transaction costs, and differentiable loss. It changes only how training
duration is selected:

1. The outer train and outer validation remain those defined by the 8A
   chronological folds.
2. The final 20% of outer-train examples, by signal date, form inner validation.
3. Inner-train examples are kept only when their `next_execution_date` is
   strictly before the first inner-validation signal. Overlapping future-return
   labels are purged.
4. An inner scaler is fitted only on the remaining inner-train features.
   Early stopping uses only the inner validation sequence.
5. A new model and optimizer are initialized with the fold seed. The model is
   trained on all outer-train examples for the selected fixed number of epochs,
   using the 8A outer scaler fitted exclusively on outer-train features.
6. The outer validation is evaluated after this final fit. Its features,
   labels, metrics, and losses do not select epochs or alter the model.

Features in the 8A fold CSVs are already outer-standardized. The audit restores
their raw scale using the saved outer-train scaler before fitting the inner
scaler. This is the same affine feature data that 8A used; the inner scaler
never sees inner-validation rows during fitting.

The configured conservative information convention remains one seven-calendar-
day lag: the feature snapshot must be dated no later than
`signal_date - 7 days`. Simulated execution occurs on the first market
observation after the Friday signal. The seven-day lag is not applied again to
the already prepared features.

## Accounting comparison

The differentiable training loss estimates costs as a fee rate multiplied by
the L1 change in foreign-currency weights from the pre-trade portfolio. It
tracks portfolio drift between periods. This smooth approximation is not the
portfolio ledger.

Validation metrics and net returns use the existing 6A/6B portfolio engine,
which determines EUR-denominated conversion notionals, charges the configured
fee per transaction, and values the resulting positions at market observations.
The audit compares both methods on controlled synthetic sequences and compares
the differentiable gross/net returns to exact outer-validation NAVs. No change
was made to portfolio accounting.

## Comparisons and diagnostics

For each outer validation, MLP V1, EUR Cash, Equal Weight Weekly, and Buy & Hold
Equal Weight use the same starting capital, price observations, daily NAV
dates, Friday signal schedule, seven-day information lag, execution dates, and
10 bps fees. Metrics are reported individually by fold. Fold CAGR averages are
not combined into a chained result: the folds reset capital and their valuation
intervals overlap at transitions, so naïve concatenation would duplicate or
omit intervals.

The audit records target and realized average allocations, maximum target
weight, currencies held, target L1 changes, near-identical weeks, effective
rebalances, turnover, annual/cumulative fees, and gross theoretical versus net
approximate/exact return. In-sample train returns are explicitly diagnostics
only.

## Scope and limitations

The final-test period begins on 2019-10-25 and is excluded from training,
epoch selection, metrics, and strategy comparison. The development boundary is
2019-10-24. The EUR and feature Parquet readers apply this date as a row filter,
so only observations through 2019-10-24 enter the in-memory market data used by
the audit. Full-file SHA-256 values are still checked against the 8A manifest
for source integrity; post-boundary values do not enter features, models,
decisions, or metrics.

Although the new nested method removes outer-validation early stopping, the
outer folds were already exposed through step 8B reports and human review. They
cannot now serve as a fully untouched final test. The FRED observations have no
certified real-time publication timestamps; the seven-day lag is a conservative
research convention, not known data availability. FRED exchange rates are not
certified executable prices. Spread, slippage, interest, and carry are not
modeled.

Run the isolated audit with:

```powershell
.\.venv\Scripts\python.exe -m forex_v1.models.mlp_training --audit --config configs\mlp_v1_audit.yaml
```

The audit writes versioned checkpoints under `data/derived/models/mlp_v1_audit/`
and creates `reports/mlp_v1_audit.md`, `reports/mlp_v1_audit.json`, and
fold-specific charts without overwriting the step 8B outputs.
