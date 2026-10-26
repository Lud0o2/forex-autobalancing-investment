# ML dataset preparation and temporal validation (step 8A)

## Scope

`forex_v1.models.dataset` prepares a supervised dataset from the existing validated
EUR-converted prices and feature Parquet files. It does **not** implement, train,
optimize, or evaluate an MLP. It does not modify the backtester or strategy APIs.

Run locally without network access:

```powershell
.\.venv\Scripts\python.exe -m forex_v1.models.dataset
```

The CLI reads `configs/ml_dataset.yaml`, `data/derived/eur_rates/`, and
`data/derived/features/`. It writes content-addressed Parquet/JSON files and fold
artifacts under `data/derived/ml/`, plus `reports/ml_dataset_validation.md`.
Paths can be overridden with `--config`, `--eur-data`, `--feature-data`,
`--output-dir`, and `--report`.

## Chronology and protected test period

The 75%/25% boundary is determined from the complete common EUR-price observation
calendar, before feature warm-up or label filtering. The first 75% of market
observations is development; the final 25% is excluded from examples, labels,
folds, and scaler fitting. This must not be changed to a split of only retained ML
rows, since doing so would move the established boundary.

Signals are calendar Fridays, including holidays. For signal date `T`, the feature
cutoff is `T - availability_lag_days` calendar days. X uses the latest actual
feature market date no later than that cutoff. It does not relabel a market date
as its publication date. FRED publication availability is unknown, so the
seven-day lag is a conservative research assumption, not a certified point-in-time
guarantee.

An execution is the first common market observation strictly after the Friday
signal, consistent with the existing backtester. Holidays are not synthesized.
Y is the simple return in each EUR-per-unit price between that execution and the
next distinct, non-colliding scheduled weekly execution. The EUR target is exactly
zero. A sample is omitted if its target endpoint would fall outside the
development partition; test-period prices are never used to complete labels.

## Inputs and targets

The default X order is currency-major (USD, GBP, JPY, CHF, CAD, AUD, NZD), then
the configured indicator order. It contains 13 existing indicators (91 columns)
and does not contain EUR technical indicators. `momentum_5/20/60` are excluded
because they duplicate the corresponding cumulative `log_return_5/20/60`
features. Raw prices and date metadata are not model inputs. Missing or non-finite
feature rows are excluded; the builder does not forward-fill or impute.

The output retains start/end prices in separate audit columns so each target
return can be recalculated. The raw dataset Parquet and JSON manifest include
separate X and Y column lists, signal/cutoff/information/execution dates,
development boundary, exclusion counts, and SHA-256 fingerprints of the input
Parquet files.

## Walk-forward folds and normalization

The default protocol is three chronological expanding-window folds, with at least
five calendar years of initial training. The validation sample ranges divide the
remaining development examples into non-overlapping consecutive blocks. For each
fold, training rows precede validation. A training row is purged if
`next_execution_date + embargo_days` is not strictly before the first validation
signal. The configured default embargo is zero days; the strict label-end purge
already prevents a training outcome from reaching validation. A positive extra
embargo can be configured.

Each fold has its own scaler, fitted only on that fold's training X. Each
currency-feature column is standardized with its training mean and population
standard deviation (`ddof=0`). Constant columns use scale 1 and therefore map to
zero. Validation values use the saved training statistics. Non-finite values are
errors rather than silently imputed. Y is not normalized.

The fold CSV exports contain dates, normalized X, and unscaled Y; their scaler
JSON records the ordered columns, means, scales, constant columns, fold dates,
purge counts, and fit scope. They are separate from the raw dataset so that each
future training run can reconstruct its own fold preprocessing without fitting
normalization on validation data.

## Limitations

These labels are research price returns, not realized execution returns. FRED
reference rates are indicative, not certified executable quotes; actual
publication availability is unknown. Spread, slippage, interest, carry, and
transaction costs are not included. The feature set is limited to the existing
indicators and must be kept causal by their existing definitions. This step does
not select a model, tune parameters against validation performance, or inspect the
final 25% test period.
