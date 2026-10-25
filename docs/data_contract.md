# Data contract

## Standardized FX observation

`forex_v1.data.schemas.FXObservation` represents one observed exchange rate with:

| Field | Required | Meaning |
| --- | --- | --- |
| `market_date` | Yes | Market calendar date (`datetime.date`), with no time component. |
| `base_currency` | Yes | ISO 4217 code for the currency being valued. |
| `quote_currency` | Yes | ISO 4217 code for the currency in which the rate is expressed. |
| `quote_per_base` | Yes | Finite, strictly positive number of quote-currency units per one base unit. |
| `source` | Yes | Non-empty provenance/source identifier. |
| `price_type` | Yes | `reference`, `close`, `bid`, or `ask`. |
| `available_at` | Conditional | Timezone-aware timestamp for when the observation was actually available. |
| `availability_known` | Yes | Explicitly states whether `available_at` is known. If false, the timestamp must be absent. |

The V1 contract currently accepts only the project currency universe: EUR, USD, GBP, JPY,
CHF, CAD, AUD, and NZD. These are ISO 4217 currency codes; currencies outside this V1
universe are rejected by the schema.

## Quotation convention

The rate is always `quote_per_base`: quote currency per one base currency. Thus
`EUR/USD = 1.08` is represented by `base_currency="EUR"`, `quote_currency="USD"` and
`quote_per_base=1.08`, meaning one EUR equals 1.08 USD. `USD/EUR` is a different pair and
must be represented with the base and quote currencies reversed; rates are not silently
inverted.

Market date and availability timestamp are separate fields and types. The availability
timestamp must carry a timezone. If the actual publication/availability time is unknown,
use `availability_known=False` and `available_at=None`; do not infer a time from the market
date.

## Reference and executable prices

A `reference` or `close` observation does not by itself establish that a trade could have
executed at that rate. Bid/ask observations are distinct price types. FRED H.10 rates are
indicative reference observations, not OHLC candles or proof of an executable bid/ask
market. Do not manufacture Open, High, Low, Bid, Ask, or Volume fields when the source does
not provide them.

## Missing values and duplicate observations

An `FXObservation` always has a real positive finite rate. Missing source values are not
represented as zero, NaN, infinity, or a fabricated observation; retain them as missing or
absent in the source-specific representation and do not silently fill them. Different
currencies may have different historical coverage and trading calendars. No common
calendar, interpolation, or forward-fill behavior is implied by this contract.

`validate_observations` rejects duplicates for the same source, base currency, quote
currency, market date, and price type. Different sources or price types remain separate
observations.

## Temporal availability and look-ahead prevention

Consumers that make decisions at time `t` must use only observations whose actual
availability is known to be no later than `t`. When availability is unknown, this data
contract does not assert that the observation was available at its market date; callers
must apply an explicit conservative policy rather than inventing a timestamp. The
observation's market date remains unchanged regardless of its publication time.

## Local storage layout

- `data/raw/fred/<SERIES_ID>/<SERIES_ID>_<retrieval-time>_<sha256>.json`: canonical complete
  FRED response body, including missing-value markers.
- The adjacent `.manifest.json` records the response SHA-256, series, safe request
  parameters, UTC retrieval time, and available FRED metadata. It does not contain the API
  key or authentication headers.
- `data/processed/fred/<SERIES_ID>/<SERIES_ID>_<content-sha256>.parquet`: sorted,
  standardized observations with schema version and raw-file reference.

Raw response bodies and manifests are immutable. Reprocessing the same raw response
produces the same content-addressed Parquet path; corrected source data creates a new
processed version and does not replace the previous one.

If a raw provider supplies a Parquet file, preserve that original file under `data/raw/`;
do not replace it with the standardized dataset. Any regeneration of processed output must
be an explicit operation rather than a silent overwrite. Use
`forex_v1.data.storage.parquet_store.rebuild_from_raw(manifest_path, processed_root)` to
verify a raw response and reconstruct its processed Parquet snapshot without network access.
Use `read_raw_response` to verify the raw-body hash and `read_parquet` to load and validate a
processed snapshot.

`forex_v1.data.layout` exposes the default directory paths. This step defines no Parquet
writer, ETL pipeline, cache, or overwrite mechanism; those behaviors must be implemented
explicitly in a later task.
