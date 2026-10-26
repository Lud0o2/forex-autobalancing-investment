# Backtest performance evaluation (step 7A)

## Scope and inputs

`forex_v1.evaluation` reads existing dated NAV, transaction, and realized-allocation
exports. It does not simulate orders, select a strategy, or depend on a strategy or
model implementation. The analysis can be run on a complete history or an inclusive
date range:

```powershell
.\.venv\Scripts\python.exe -m forex_v1.evaluation.reporting
.\.venv\Scripts\python.exe -m forex_v1.evaluation.reporting `
  --start-date 2010-01-01 --end-date 2019-12-31
```

The CLI defaults to `reports/backtest_nav.csv`,
`reports/backtest_transactions.csv`, `reports/backtest_allocations.csv`,
`reports/backtest_validation.md`, and `configs/backtest.yaml`. Source SHA-256
hashes are embedded in `reports/performance_metrics.json` for traceability.

## Metric conventions

- **Total return**: `NAV_last / NAV_first - 1`.
- **CAGR**: `(NAV_last / NAV_first) ** (1 / elapsed_calendar_years) - 1`.
  Elapsed years use ACT/ACT calendar fractions, including leap years, between the
  first and last valid observation. A single-observation series has no defined
  CAGR.
- **Maximum drawdown**: minimum of `NAV / running_peak - 1`. It is zero when no
  decline from a prior peak occurred.
- **Observation returns**: simple percentage changes between consecutive valid NAV
  observations. No missing market observations are synthesized.
- **Annualized volatility**: sample standard deviation (`ddof=1`) of observation
  returns multiplied by `sqrt(252)` by default. 252 observations per year is an
  approximation; the source calendar can have holidays and gaps.
- **Sharpe**: mean observation return divided by sample standard deviation and
  multiplied by `sqrt(252)`. Risk-free return is explicitly zero because cash is
  not remunerated in this V1 model.
- **Sortino**: mean observation return divided by downside RMS, annualized by
  `sqrt(252)`. Downside RMS is `sqrt(mean(min(return, 0) ** 2))`; zero-return
  observations contribute zero downside.
- **Calmar**: CAGR divided by the absolute maximum drawdown. It is undefined if no
  drawdown or CAGR is undefined.
- **Best/worst year and month**: calendar period close-to-close returns. The
  baseline is the previous observed NAV; for the first available period it is the
  first NAV. A period truncated by the input's first/last boundary is marked
  `partial` in period exports and annual reporting.
- **Win rate**: fraction of positive observation-to-observation returns; zero
  returns are not wins.
- **Turnover**: gross sum of executed EUR notionals divided by average NAV over the
  analysis interval. Buys and sells both count, so this is gross traded turnover,
  not half-turnover.
- **Total fees**: sum of transaction `fee_eur` values with execution dates inside
  the analyzed NAV interval. No transactions means zero fees and zero turnover.

Undefined metrics are represented as JSON `null` and Markdown `N/D`; infinity is
never used to represent a zero denominator. Missing NAV values are dropped and
counted. Duplicate dates, non-finite values, non-positive NAV, and malformed
transaction inputs are rejected.

## Period selection and model test boundary

The report includes the full selected history and descriptive windows 2000–2009,
2010–2019, and 2020–2026, plus an optional inclusive custom date range. These
comparisons diagnose a fixed reference scenario and the simulation engine only.
They must not be used to optimize a future model on the final 25% test period.
Maintain the project's chronological 75% development / 25% final-test boundary;
this module does not fit scalers, choose hyperparameters, or change that split.

`as_of` can also be passed to `calculate_metrics` / `analyze_performance` to
exclude later NAV observations before calculating any metric, useful for causal
cutoff checks.

## Visualizations and report

The evaluation plots are generated from the same filtered NAV, transaction, and
allocation records as the metric tables:

1. Capital curve (EUR).
2. Drawdown curve (percentage).
3. Annual calendar returns.
4. Monthly return heatmap.
5. Realized currency weights over time.
6. Cumulative transaction fees.

The CLI writes `reports/performance_analysis.md`, a JSON comparison-friendly
`reports/performance_metrics.json`, and six `performance_*.png` charts. A chart's
measurements follow the same interval filters as the metrics. The source backtest
exports are read-only and their hashes are retained in the JSON.

## Limitations

Metrics describe the provided NAV and trade exports; they do not certify the
underlying prices as executable. The current FRED H.10 history has unknown
point-in-time publication availability, uses indicative reference rates, and does
not model spread, slippage, tax, interest, or carry. Annualization assumes the
configured number of observations per year and should not be treated as a
statistical guarantee for irregular observation intervals.
