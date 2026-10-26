# Technical features

## Scope and input

The feature pipeline reads the validated EUR-denominated Parquet produced by
`forex_v1.data.eur_conversion`. It calculates indicators for USD, GBP, JPY, CHF,
CAD, AUD, and NZD; EUR is cash at a constant value of one and is intentionally
excluded from technical indicators.

Run it from the repository root:

```powershell
.\.venv\Scripts\python.exe -m forex_v1.features.builder --help
.\.venv\Scripts\python.exe -m forex_v1.features.builder
```

The console-script equivalent is `forex-v1-features`. `--eur-parquet` accepts
either one validated Parquet file or a directory containing exactly one such
snapshot. Other paths can be set with `--config`, `--output-root`, and `--report`.
The pipeline performs no network access.

## Causality and calendar

Each indicator uses only the current and earlier observations in its currency
series. Rolling windows are trailing windows of market observations, not calendar
days. The shared input calendar is retained: the EUR conversion stage already
restricts it to dates with same-day observations for all seven FRED series. This
module does not add dates, forward-fill, backfill, interpolate, or align to future
observations.

The first observations that lack sufficient history remain null in the output.
Missingness is represented both by nullable feature values and the row fields
`missing_feature_count` and `all_features_available`. Original market dates,
source Parquet name/hash, raw source identifiers, feature configuration version,
and feature configuration hash are retained.

FRED historical publication availability is not certified. Feature rows always
retain `available_at=None` and `availability_known=False`. They are research
features, not point-in-time-certified inputs or executable market prices. The
pipeline does not fit any global scaler; a future model must fit scaling
transformations on each training fold only.

## Indicators

For a positive EUR-per-unit price series `P[t]` and an observation window `n`:

| Feature | Formula and windows | Output convention |
| --- | --- | --- |
| Log return | `ln(P[t] / P[t-n])`, `n = 1, 5, 20, 60` | Signed, unit-free change |
| Momentum | Cumulative log return over `n = 5, 20, 60` observations | `ln(P[t] / P[t-n])`; signed and unit-free |
| Relative SMA | `P[t] / mean(P[t-n+1:t]) - 1`, `n = 20, 50, 200` | Signed relative distance |
| RSI | Rolling simple average of gains divided by average gains plus losses, `n = 14` | Scaled to `[0, 1]` |
| Volatility | Sample standard deviation of one-observation log returns, `n = 20, 60` | Annualized by `sqrt(252)` observations |
| Bollinger z-score | `(P[t] - rolling_mean) / rolling_population_std`, `n = 20` | Signed, dimensionless deviation |
| Channel position | `(P[t] - rolling_min) / (rolling_max - rolling_min)`, `n = 20` | Bounded to `[0, 1]` |
| Drawdown | `P[t] / rolling_max(P, n=60) - 1` | Non-positive relative decline |

Both `log_return_n` and `momentum_n` use the cumulative log-return formula. The
overlap is intentional in the default configuration so that short-horizon returns
and the named momentum family can be selected independently. Disable either
family in YAML if a downstream experiment should not include both.

The configured volatility factor annualizes by 252 observations. As observations
may be separated by more than one calendar day, this convention is not an
annualization adjusted for irregular time intervals; the separate EUR source data
retains the observation dates and interval lengths.

## Edge-case conventions

- No feature value is emitted until its full trailing window is available.
- A constant RSI window is `0.5`; a gain-only window is `1`, and a loss-only
  window is `0`.
- A zero-standard-deviation Bollinger window has z-score `0`.
- A zero-width channel window has position `0.5`.
- Constant-series volatility and drawdown are `0` once their windows are ready.
- Prices must be finite, non-null, and strictly positive. Invalid input raises an
  error instead of silently producing artificial values.
- Non-finite feature values are rejected before storage.

## Configuration and adding indicators

`configs/features.yaml` is the versioned configuration. Each indicator has an
`enabled` boolean and a non-empty list of positive observation windows. Volatility
and Bollinger windows must contain at least two observations; other indicators
require at least one. The annualization observation count is configurable.

To add an indicator:

1. Implement an independent trailing calculation in
   `src/forex_v1/features/technical.py`, with explicit input validation and edge
   conventions.
2. Add its identifier to `SUPPORTED_INDICATORS` and its minimum valid window in
   `src/forex_v1/features/validation.py`.
3. Register the function in `calculate_technical_features`.
4. Add formula, known-value, warm-up, constant/zero-denominator, scale-invariance,
   and no-look-ahead tests.
5. Enable/configure it in YAML and document its formula here.

Changing the normalized configuration changes its SHA-256 provenance. Output
Parquet files are content-addressed and immutable; repeated identical runs reuse
the same dataset rather than overwriting a prior result.

## Outputs

- `data/derived/features/features_v1_<sha256>.parquet`: seven currencies in
  date/currency long format, indicator values and provenance metadata.
- `reports/features_validation.md`: actual row/date counts, per-feature valid and
  missing counts, ranges, first complete date across the seven currencies, and
  the point-in-time limitation.

The Parquet reader validates the schema and version metadata, content SHA-256,
feature missingness flags, chronological ordering, finite values, and source and
configuration provenance.
