# EUR-denominated currency rates

## Scope and input selection

This step transforms the seven verified FRED H.10 series into a common-calendar
research dataset. It does not create executable broker prices, interest accrual,
transaction costs, signals, strategies, or backtests.

The local pipeline reads only `data/processed/fred/<SERIES_ID>/*.parquet` and their
referenced raw JSON manifests. It verifies Parquet schema/content hashes, raw JSON
SHA-256 hashes, the series and raw-file provenance, the FRED count/offset metadata,
and that the processed rows reproduce from the raw response. Eligible inputs must
have a request window beginning on 1999-01-01 and ending in 2026 or later, and
observations spanning 1999 through 2026. The 2024 EUR/USD pilot is therefore
ineligible.

When several eligible snapshots exist, selection is deterministic: latest requested
end date, then greatest valid observation count, then lexicographically greatest
content-addressed Parquet filename. No network access is used.

Run the local conversion and report generation from the repository root:

```powershell
.\.venv\Scripts\python.exe -m forex_v1.data.eur_conversion
```

Optional `--processed-root`, `--raw-root`, `--derived-root`, and `--reports-dir`
arguments allow alternate local directories. The result is an immutable Parquet
snapshot under `data/derived/eur_rates/`, with a SHA-256 content identifier and
transformation/schema versions. Repeating the same conversion reuses the same
snapshot. Source files are read-only inputs and are never rewritten.

## Quote directions and EUR conversions

FRED rates are quoted as units of the quote currency per one base currency:

| Series | Meaning |
| --- | --- |
| `DEXUSEU` | USD per EUR |
| `DEXUSUK` | USD per GBP |
| `DEXJPUS` | JPY per USD |
| `DEXSZUS` | CHF per USD |
| `DEXCAUS` | CAD per USD |
| `DEXUSAL` | USD per AUD |
| `DEXUSNZ` | USD per NZD |

The output field `eur_per_unit` means euros for one unit of the row's currency:

```text
EUR = 1
USD = 1 / DEXUSEU
GBP = DEXUSUK / DEXUSEU
JPY = 1 / (DEXJPUS * DEXUSEU)
CHF = 1 / (DEXSZUS * DEXUSEU)
CAD = 1 / (DEXCAUS * DEXUSEU)
AUD = DEXUSAL / DEXUSEU
NZD = DEXUSNZ / DEXUSEU
```

All inputs to a conversion are taken from the exact same `market_date`. Rates must
be finite and strictly positive; zero, negative, non-finite, or overflowed derived
values are rejected. Derived rates are mathematical cross-rates from FRED reference
series, not independently observed EUR quotes and not executable prices.

## Calendar, missing observations, and availability

The output includes a date only when valid observations for all seven FRED series
exist on that same date. Candidate source dates with one or more missing component
rates are excluded and listed in the validation report. Original market dates are
retained in `market_date` and `source_market_dates`. No forward-fill, interpolation,
or future observation is used to complete a date.

FRED H.10 publication availability is not certified historically. Derived rows keep
`available_at=None` and `availability_known=False`; the EUR-denominated history is a
research dataset, not a point-in-time-certified or executable-price history.

## Returns and gaps

For consecutive common observations:

```text
simple_return_t = eur_per_unit_t / eur_per_unit_previous - 1
log_return_t = ln(eur_per_unit_t / eur_per_unit_previous)
```

The first return for each currency is null. EUR remains exactly 1 EUR per EUR and
has zero simple/log return after its first observation. The dataset also stores the
previous market date and `calendar_days_since_previous`. Returns spanning multiple
calendar days are changes between observations over that recorded interval; they
must not be interpreted as one-calendar-day returns. Interest, carry, and conversion
fees are not included.

## Outputs and validation

- `data/derived/eur_rates/eur_rates_v1_<sha256>.parquet`: long-format rows with date,
  currency, EUR value, simple/log return, source identifiers and dates, availability
  status, and version metadata.
- `reports/eur_conversion_validation.md`: selected source windows and hashes, common
  period, excluded dates, first/last values, return ranges, anomalies, and limitations.
- `reports/eur_conversion_index.png`: each currency's EUR-per-unit series normalized
  to 100 on the first common date. This is a relative-value visualization only.

The Parquet reader verifies the schema and transformation versions, content hash,
positive finite valuations, chronological order, complete eight-currency rows,
same-date source provenance, unknown availability, EUR invariants, and return
intervals. The generated data and reports are local outputs and are excluded from Git.
