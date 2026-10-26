# FRED Forex data source

## Provider and series

The connector uses the official FRED Series Observations API:

<https://fred.stlouisfed.org/docs/api/fred/series_observations.html>

Each series is requested independently. The rate convention is the exact FRED pair
direction shown below: units of `quote_currency` per one `base_currency`. The connector
does not invert series or derive additional pairs.

| FRED series | Base | Quote | Interpretation |
| --- | --- | --- | --- |
| `DEXUSEU` | EUR | USD | USD per EUR |
| `DEXUSUK` | GBP | USD | USD per GBP |
| `DEXJPUS` | USD | JPY | JPY per USD |
| `DEXSZUS` | USD | CHF | CHF per USD |
| `DEXCAUS` | USD | CAD | CAD per USD |
| `DEXUSAL` | AUD | USD | USD per AUD |
| `DEXUSNZ` | NZD | USD | USD per NZD |

Official series pages:

- [DEXUSEU](https://fred.stlouisfed.org/series/DEXUSEU)
- [DEXUSUK](https://fred.stlouisfed.org/series/DEXUSUK)
- [DEXJPUS](https://fred.stlouisfed.org/series/DEXJPUS)
- [DEXSZUS](https://fred.stlouisfed.org/series/DEXSZUS)
- [DEXCAUS](https://fred.stlouisfed.org/series/DEXCAUS)
- [DEXUSAL](https://fred.stlouisfed.org/series/DEXUSAL)
- [DEXUSNZ](https://fred.stlouisfed.org/series/DEXUSNZ)

Each converted observation is marked `reference` and its provenance is
`fred_h10:<SERIES_ID>`.

## API key

The API key is read only from the process environment variable `FRED_API_KEY`. Do not put
the key in Python source, YAML, tests, logs, or committed files. On Windows, configure
`FRED_API_KEY` in the user environment (outside the repository), then start a new VS Code
or PowerShell process so it inherits the variable.

Verify the variable is present without printing its value:

```powershell
if ([string]::IsNullOrWhiteSpace($env:FRED_API_KEY)) {
    throw "FRED_API_KEY is not configured for this PowerShell process."
}
```

The client accepts optional observation start and end dates, uses a bounded HTTP timeout,
and retries a limited number of times for network failures and temporary HTTP statuses.
The connector does not download data on import; a request is made only when
`FredClient.fetch_series` is called. Supply a test/dummy HTTP session in unit tests to keep
them offline.

## Limitations and temporal availability

FRED H.10 exchange-rate observations are indicative reference rates, not executable bid/ask
quotes and not OHLC candles. The connector produces no Open, High, Low, Bid, Ask, or Volume
fields.

The observations are daily, while official publication is generally weekly. The API's
observation date is a market/reference date; it does not establish when that value became
available to a historical decision. Vintage metadata such as `realtime_start` and
`realtime_end` is not treated as a precise publication timestamp. The connector therefore
sets `availability_known=False` and `available_at=None` for every result. It never assumes
same-day availability or invents a release time. A later task must define a conservative
availability policy for time-sensitive research.

Missing FRED values (`.` or empty strings) are skipped as absent observations. Zero,
negative, non-finite, and non-numeric rates are rejected. No missing days are filled or
interpolated. Different series may have different histories and calendars.

## Local persistence and reconstruction

When a caller explicitly passes a `FredRawResponse` to
`forex_v1.data.storage.raw_store.write_raw_response`, the full JSON response is stored
under `data/raw/fred/<SERIES_ID>/` with an adjacent manifest. The manifest records the
SHA-256 hash of the canonical JSON body, UTC retrieval time, sanitized request parameters,
and FRED metadata such as `count`, `offset`, `limit`, `realtime_start`, and `realtime_end`
when supplied. Query credentials and authentication headers are never persisted.

`forex_v1.data.storage.parquet_store.rebuild_from_raw` verifies the raw manifest/body,
converts the stored response with the existing FRED parser, and writes a versioned file
under `data/processed/fred/<SERIES_ID>/`. Processed rows include the `raw_file_id`, series
identifier, schema version, pair, market date, rate, price type, and availability fields.
The same raw snapshot produces the same processed path; a changed response has a different
content-addressed version and does not overwrite prior files.

For local verification/reconstruction:

```powershell
.\.venv\Scripts\python.exe -m pytest -q tests\test_storage.py
```

No data is fetched by this storage API. A later download workflow must explicitly call the
connector, store the returned raw response, then invoke the rebuild function.

## Offline verification

From the repository root, run the suite using the project environment:

```powershell
.\.venv\Scripts\python.exe -m pytest -q
.\.venv\Scripts\python.exe -m ruff check .
```

The provider tests use fake responses and an injected fake HTTP session; they do not send
network requests and do not need a real API key.

## Explicit import and coverage workflow

The `forex-v1-import` command requires an explicit mode. A dry run is offline and does not
create raw or processed market-data files:

```powershell
.\.venv\Scripts\python.exe -m forex_v1.data.cli --dry-run
```

To authorize a real request, provide the FRED key through `FRED_API_KEY` and use
`--execute`. The default request is the required pilot (`DEXUSEU`, 2024-01-01 through
2024-12-31); series and dates can be selected explicitly:

```powershell
.\.venv\Scripts\python.exe -m forex_v1.data.cli --execute --series DEXUSEU --start-date 2024-01-01 --end-date 2024-12-31
```

Only after the 2024 pilot succeeds, the full-history option requests all seven configured
series from 1999-01-01 through the last complete UTC day:

```powershell
.\.venv\Scripts\python.exe -m forex_v1.data.cli --execute --full-history
```

Each completed response is preserved in `data/raw/fred/<SERIES_ID>/`, then validated and
reconstructed into a versioned Parquet file in `data/processed/fred/<SERIES_ID>/`.
Paginated requests are combined only when the declared FRED count is fully received.
Errors for one series are reported while the remaining selected series are attempted.
The workflow writes `reports/data_coverage.json` and
`reports/data_coverage.md`; these are local generated reports and are ignored by Git.
They include raw content hashes and raw/processed file identifiers for provenance.
Each retrieval remains a distinct immutable raw snapshot. Reprocessing identical
standardized observations reuses the existing Parquet snapshot; if reused, the report and
CLI show the original raw file identifier referenced by that Parquet file as well as the
current retrieval's raw identifier.
The reported coverage percentage is the number of valid observations divided by the
number of records returned by FRED (including explicit missing values and any duplicate
records); weekday-gap counts are reported separately and are not a holiday calendar.
The weekday-gap counts are descriptive: the API response alone cannot reliably separate
public holidays from genuine coverage gaps. Common-history metrics use the exact
intersection of valid dates and do not forward-fill or interpolate.

No historical `available_at` timestamp is inferred during import. Even after a successful
download, these H.10 observations are not a point-in-time-certified history for executable
decisions. Verify local orchestration and report behavior without network access using:

```powershell
.\.venv\Scripts\python.exe -m pytest -q tests\test_importer.py tests\test_fred_provider.py
```
