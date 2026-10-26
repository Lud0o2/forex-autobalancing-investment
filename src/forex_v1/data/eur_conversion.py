"""Build reproducible, date-aligned FX rates expressed in EUR."""

import argparse
import hashlib
import json
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import pyarrow as pa
import pyarrow.parquet as pq

from forex_v1.data.layout import PROJECT_ROOT
from forex_v1.data.providers.fred import (
    FRED_FX_SERIES,
    parse_fred_observations,
)
from forex_v1.data.schemas import SUPPORTED_CURRENCIES
from forex_v1.data.storage._files import ImmutableFileError, write_immutable_file
from forex_v1.data.storage.parquet_store import (
    PARQUET_SCHEMA_VERSION as FRED_PARQUET_SCHEMA_VERSION,
)
from forex_v1.data.storage.parquet_store import (
    ParquetSnapshot,
    ParquetStoreError,
    read_parquet,
)
from forex_v1.data.storage.raw_store import (
    RawFredSnapshot,
    RawStoreError,
    read_raw_response,
)

TRANSFORM_VERSION = "1"
EUR_DATASET_SCHEMA_VERSION = 1
EXPECTED_REQUEST_START = date(1999, 1, 1)
EXPECTED_REQUEST_END_YEAR = 2026
SERIES_BY_CURRENCY: Mapping[str, tuple[str, ...]] = {
    "EUR": (),
    "USD": ("DEXUSEU",),
    "GBP": ("DEXUSEU", "DEXUSUK"),
    "JPY": ("DEXUSEU", "DEXJPUS"),
    "CHF": ("DEXUSEU", "DEXSZUS"),
    "CAD": ("DEXUSEU", "DEXCAUS"),
    "AUD": ("DEXUSEU", "DEXUSAL"),
    "NZD": ("DEXUSEU", "DEXUSNZ"),
}
CURRENCY_ORDER = ("EUR", "USD", "GBP", "JPY", "CHF", "CAD", "AUD", "NZD")
CONTENT_SHA256_KEY = b"forex_v1.eur_rates.content_sha256"
SCHEMA_VERSION_KEY = b"forex_v1.eur_rates.schema_version"
TRANSFORM_VERSION_KEY = b"forex_v1.eur_rates.transform_version"

EUR_RATES_SCHEMA = pa.schema(
    [
        pa.field("market_date", pa.date32(), nullable=False),
        pa.field("currency", pa.string(), nullable=False),
        pa.field("eur_per_unit", pa.float64(), nullable=False),
        pa.field("simple_return", pa.float64(), nullable=True),
        pa.field("log_return", pa.float64(), nullable=True),
        pa.field("previous_market_date", pa.date32(), nullable=True),
        pa.field("calendar_days_since_previous", pa.int32(), nullable=True),
        pa.field("source_series_ids", pa.list_(pa.string()), nullable=False),
        pa.field("source_raw_file_ids", pa.list_(pa.string()), nullable=False),
        pa.field("source_parquet_files", pa.list_(pa.string()), nullable=False),
        pa.field("source_market_dates", pa.list_(pa.date32()), nullable=False),
        pa.field("available_at", pa.timestamp("us", tz="UTC"), nullable=True),
        pa.field("availability_known", pa.bool_(), nullable=False),
    ]
)


class EurConversionError(RuntimeError):
    """Raised when source histories cannot safely produce EUR-denominated rates."""


@dataclass(frozen=True, slots=True)
class SelectedSource:
    """One verified full-history processed file and its verified raw source."""

    series_id: str
    parquet: ParquetSnapshot
    raw: RawFredSnapshot
    request_start: date
    request_end: date


@dataclass(frozen=True, slots=True)
class EurConversionResult:
    """Complete derived dataset and source metadata used to create it."""

    sources: Mapping[str, SelectedSource]
    rows: tuple[dict[str, Any], ...]
    common_dates: tuple[date, ...]
    excluded_dates: tuple[date, ...]
    derived_path: Path | None = None


@dataclass(frozen=True, slots=True)
class EurParquetSnapshot:
    """Validated EUR-rate dataset loaded from its immutable Parquet file."""

    rows: tuple[dict[str, Any], ...]
    transform_version: str
    schema_version: int
    path: Path


def convert_rates_to_eur(rates_by_series: Mapping[str, float]) -> dict[str, float]:
    """Convert the seven exact FRED quote conventions into EUR per currency unit."""
    expected = set(FRED_FX_SERIES)
    supplied = set(rates_by_series)
    if supplied != expected:
        missing = sorted(expected - supplied)
        unexpected = sorted(supplied - expected)
        raise ValueError(
            f"Expected all seven FRED rates; missing={missing}, unexpected={unexpected}"
        )

    rates: dict[str, float] = {}
    for series_id, raw_rate in rates_by_series.items():
        if isinstance(raw_rate, bool) or not isinstance(raw_rate, (int, float)):
            raise TypeError(f"{series_id} must be a finite positive number")
        value = float(raw_rate)
        if not math.isfinite(value) or value <= 0:
            raise ValueError(f"{series_id} must be finite and strictly positive")
        rates[series_id] = value

    eur_usd = rates["DEXUSEU"]
    denominators = {
        "JPY": rates["DEXJPUS"] * eur_usd,
        "CHF": rates["DEXSZUS"] * eur_usd,
        "CAD": rates["DEXCAUS"] * eur_usd,
    }
    for currency, denominator in denominators.items():
        if not math.isfinite(denominator) or denominator <= 0:
            raise ValueError(f"Derived denominator for {currency} is not finite and positive")
    converted = {
        "EUR": 1.0,
        "USD": 1.0 / eur_usd,
        "GBP": rates["DEXUSUK"] / eur_usd,
        "JPY": 1.0 / denominators["JPY"],
        "CHF": 1.0 / denominators["CHF"],
        "CAD": 1.0 / denominators["CAD"],
        "AUD": rates["DEXUSAL"] / eur_usd,
        "NZD": rates["DEXUSNZ"] / eur_usd,
    }
    for currency, value in converted.items():
        if not math.isfinite(value) or value <= 0:
            raise ValueError(f"Derived EUR rate for {currency} is not finite and positive")
    return converted


def _request_dates(raw: RawFredSnapshot) -> tuple[date, date]:
    start = raw.request_params.get("observation_start")
    end = raw.request_params.get("observation_end")
    if not isinstance(start, str) or not isinstance(end, str):
        raise EurConversionError(
            f"{raw.series_id} raw manifest does not record both requested date bounds"
        )
    try:
        return date.fromisoformat(start), date.fromisoformat(end)
    except ValueError as error:
        raise EurConversionError(
            f"{raw.series_id} raw manifest contains an invalid request date"
        ) from error


def _load_source(
    parquet_path: Path,
    *,
    raw_root: Path,
    series_id: str,
    require_full_history: bool = True,
) -> SelectedSource:
    try:
        parquet = read_parquet(parquet_path)
    except ParquetStoreError as error:
        raise EurConversionError(
            f"Cannot validate source Parquet {parquet_path}: {error}"
        ) from error

    if parquet.series_id != series_id or not parquet.raw_file_ids:
        raise EurConversionError(
            f"Source Parquet has inconsistent series provenance: {parquet_path}"
        )
    if len(set(parquet.raw_file_ids)) != 1:
        raise EurConversionError(f"Source Parquet references multiple raw files: {parquet_path}")

    raw_file_id = parquet.raw_file_ids[0]
    raw_path = raw_root / Path(raw_file_id)
    manifest_path = raw_path.with_suffix(".manifest.json")
    try:
        raw = read_raw_response(manifest_path)
    except RawStoreError as error:
        raise EurConversionError(f"Cannot verify raw source for {parquet_path}: {error}") from error
    if raw.series_id != series_id or raw.raw_file_id != raw_file_id:
        raise EurConversionError(f"Raw manifest does not match source Parquet {parquet_path}")

    request_start, request_end = _request_dates(raw)
    if require_full_history and (
        request_start != EXPECTED_REQUEST_START or request_end.year < EXPECTED_REQUEST_END_YEAR
    ):
        raise EurConversionError(
            f"{parquet_path} is not a full-history snapshot "
            f"(requested {request_start} through {request_end})"
        )

    observations = raw.payload.get("observations")
    count = raw.fred_metadata.get("count")
    try:
        declared_count = int(count)
    except (TypeError, ValueError) as error:
        raise EurConversionError(f"{series_id} raw response has no valid FRED count") from error
    if not isinstance(observations, list) or declared_count != len(observations):
        raise EurConversionError(f"{series_id} raw response is incomplete")
    if int(raw.fred_metadata.get("offset", 0)) != 0:
        raise EurConversionError(f"{series_id} raw response does not begin at offset zero")

    rebuilt = parse_fred_observations(
        series_id,
        raw.payload,
        start_date=request_start,
        end_date=request_end,
    )
    if rebuilt != parquet.observations:
        raise EurConversionError(
            f"{series_id} Parquet rows do not reproduce from the referenced raw response"
        )
    first_date = min(item.market_date for item in parquet.observations)
    last_date = max(item.market_date for item in parquet.observations)
    if require_full_history and (
        first_date.year != EXPECTED_REQUEST_START.year or last_date.year < EXPECTED_REQUEST_END_YEAR
    ):
        raise EurConversionError(
            f"{series_id} does not cover the expected 1999–2026 history "
            f"({first_date} through {last_date})"
        )
    return SelectedSource(
        series_id=series_id,
        parquet=parquet,
        raw=raw,
        request_start=request_start,
        request_end=request_end,
    )


def select_complete_snapshots(
    processed_root: str | Path,
    raw_root: str | Path,
) -> dict[str, SelectedSource]:
    """Select the latest verified 1999–2026 source snapshot per FRED series.

    Selection order is deterministic: latest requested end date, then most valid rows,
    then lexicographically greatest content-addressed Parquet filename.
    """
    processed_root = Path(processed_root)
    raw_root = Path(raw_root)
    selected: dict[str, SelectedSource] = {}
    for series_id in FRED_FX_SERIES:
        series_directory = processed_root / "fred" / series_id
        if not series_directory.is_dir():
            raise EurConversionError(f"Missing processed source directory for {series_id}")
        candidates: list[SelectedSource] = []
        for parquet_path in sorted(series_directory.glob("*.parquet")):
            source = _load_source(
                parquet_path,
                raw_root=raw_root,
                series_id=series_id,
                require_full_history=False,
            )
            first_date = min(item.market_date for item in source.parquet.observations)
            last_date = max(item.market_date for item in source.parquet.observations)
            if (
                source.request_start != EXPECTED_REQUEST_START
                or source.request_end.year < EXPECTED_REQUEST_END_YEAR
                or first_date.year != EXPECTED_REQUEST_START.year
                or last_date.year < EXPECTED_REQUEST_END_YEAR
            ):
                continue
            candidates.append(source)
        if not candidates:
            raise EurConversionError(f"No verified processed snapshots found for {series_id}")
        selected[series_id] = max(
            candidates,
            key=lambda item: (
                item.request_end,
                len(item.parquet.observations),
                item.parquet.path.name,
            ),
        )
    return selected


def build_eur_conversion(
    sources: Mapping[str, SelectedSource],
) -> EurConversionResult:
    """Create common-calendar EUR valuations and interval-aware returns."""
    if set(sources) != set(FRED_FX_SERIES):
        raise ValueError("EUR conversion requires all seven verified FRED source series")

    rates_by_date: dict[str, dict[date, float]] = {}
    candidate_dates: set[date] = set()
    for series_id in FRED_FX_SERIES:
        source = sources[series_id]
        if source.series_id != series_id or source.parquet.series_id != series_id:
            raise EurConversionError(f"Selected source key does not match {series_id}")
        rates_by_date[series_id] = {
            observation.market_date: observation.quote_per_base
            for observation in source.parquet.observations
        }
        raw_observations = source.raw.payload.get("observations")
        if not isinstance(raw_observations, list):
            raise EurConversionError(f"{series_id} raw response has no observations list")
        for item in raw_observations:
            if not isinstance(item, Mapping) or not isinstance(item.get("date"), str):
                raise EurConversionError(f"{series_id} raw response has an invalid market date")
            try:
                candidate_dates.add(date.fromisoformat(item["date"]))
            except ValueError as error:
                raise EurConversionError(
                    f"{series_id} raw response has an invalid market date"
                ) from error

    date_sets = [set(rates_by_date[series_id]) for series_id in FRED_FX_SERIES]
    common_dates = tuple(sorted(set.intersection(*date_sets)))
    if not common_dates:
        raise EurConversionError("The seven verified source series have no common market dates")
    excluded_dates = tuple(sorted(candidate_dates - set(common_dates)))

    rows: list[dict[str, Any]] = []
    previous_date: date | None = None
    previous_values: dict[str, float] = {}
    for market_date in common_dates:
        source_rates = {
            series_id: rates_by_date[series_id][market_date] for series_id in FRED_FX_SERIES
        }
        values = convert_rates_to_eur(source_rates)
        interval_days = (market_date - previous_date).days if previous_date is not None else None
        for currency in CURRENCY_ORDER:
            value = values[currency]
            simple_return = None
            log_return = None
            if previous_date is not None:
                simple_return = value / previous_values[currency] - 1.0
                log_return = math.log(value / previous_values[currency])
                if not math.isfinite(simple_return) or not math.isfinite(log_return):
                    raise EurConversionError(f"Non-finite return for {currency} on {market_date}")
            series_ids = SERIES_BY_CURRENCY[currency]
            rows.append(
                {
                    "market_date": market_date,
                    "currency": currency,
                    "eur_per_unit": value,
                    "simple_return": simple_return,
                    "log_return": log_return,
                    "previous_market_date": previous_date,
                    "calendar_days_since_previous": interval_days,
                    "source_series_ids": list(series_ids),
                    "source_raw_file_ids": [
                        sources[series_id].raw.raw_file_id for series_id in series_ids
                    ],
                    "source_parquet_files": [
                        sources[series_id].parquet.path.name for series_id in series_ids
                    ],
                    "source_market_dates": [market_date for _ in series_ids],
                    "available_at": None,
                    "availability_known": False,
                }
            )
        previous_date = market_date
        previous_values = values

    return EurConversionResult(
        sources=dict(sources),
        rows=tuple(rows),
        common_dates=common_dates,
        excluded_dates=excluded_dates,
    )


def _canonical_rows(rows: Sequence[Mapping[str, Any]]) -> bytes:
    serializable: list[dict[str, Any]] = []
    for record in rows:
        row = dict(record)
        for field in ("market_date", "previous_market_date"):
            value = row[field]
            row[field] = value.isoformat() if isinstance(value, date) else None
        row["source_market_dates"] = [
            value.isoformat() if isinstance(value, date) else value
            for value in row["source_market_dates"]
        ]
        row["available_at"] = (
            row["available_at"].isoformat() if row["available_at"] is not None else None
        )
        serializable.append(row)
    try:
        return json.dumps(
            serializable,
            allow_nan=False,
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("utf-8")
    except (TypeError, ValueError) as error:
        raise EurConversionError("Derived EUR rows cannot be serialized safely") from error


def write_eur_parquet(
    derived_root: str | Path,
    rows: Sequence[Mapping[str, Any]],
) -> EurParquetSnapshot:
    """Write an immutable, content-addressed Parquet dataset and verify its readback."""
    if not rows:
        raise EurConversionError("Refusing to write an empty EUR-rate dataset")
    body = _canonical_rows(rows)
    content_digest = hashlib.sha256(body).hexdigest()
    output_path = (
        Path(derived_root)
        / "eur_rates"
        / f"eur_rates_v{TRANSFORM_VERSION}_{content_digest}.parquet"
    )
    metadata = {
        CONTENT_SHA256_KEY: content_digest.encode("ascii"),
        SCHEMA_VERSION_KEY: str(EUR_DATASET_SCHEMA_VERSION).encode("ascii"),
        TRANSFORM_VERSION_KEY: TRANSFORM_VERSION.encode("ascii"),
    }
    table = pa.Table.from_pylist(
        [dict(row) for row in rows],
        schema=EUR_RATES_SCHEMA.with_metadata(metadata),
    )
    try:
        sink = pa.BufferOutputStream()
        pq.write_table(table, sink, compression="zstd", version="2.6")
        write_immutable_file(output_path, sink.getvalue().to_pybytes())
    except (pa.ArrowException, ImmutableFileError, OSError, TypeError, ValueError) as error:
        if isinstance(error, FileExistsError):
            raise EurConversionError(
                f"Refusing to overwrite different derived EUR data: {output_path}"
            ) from error
        raise EurConversionError(f"Could not write derived EUR Parquet: {error}") from error
    snapshot = read_eur_parquet(output_path)
    if _canonical_rows(snapshot.rows) != body:
        raise EurConversionError("Written EUR Parquet does not match its source rows")
    return snapshot


def read_eur_parquet(path: str | Path) -> EurParquetSnapshot:
    """Read a derived Parquet dataset and validate schema, digest, ordering, and values."""
    parquet_path = Path(path)
    try:
        table = pq.read_table(parquet_path)
    except (OSError, pa.ArrowException) as error:
        raise EurConversionError(
            f"Could not read derived Parquet {parquet_path}: {error}"
        ) from error
    metadata = table.schema.metadata or {}
    try:
        schema_version = int(metadata[SCHEMA_VERSION_KEY].decode("ascii"))
        transform_version = metadata[TRANSFORM_VERSION_KEY].decode("ascii")
        expected_digest = metadata[CONTENT_SHA256_KEY].decode("ascii")
    except (KeyError, UnicodeDecodeError, ValueError) as error:
        raise EurConversionError("Derived Parquet is missing valid Forex metadata") from error
    if schema_version != EUR_DATASET_SCHEMA_VERSION or transform_version != TRANSFORM_VERSION:
        raise EurConversionError("Derived Parquet schema or transformation version is unsupported")
    if not table.schema.remove_metadata().equals(EUR_RATES_SCHEMA):
        raise EurConversionError("Derived Parquet does not match the EUR-rate schema")

    rows = tuple(table.to_pylist())
    if not rows or hashlib.sha256(_canonical_rows(rows)).hexdigest() != expected_digest:
        raise EurConversionError("Derived Parquet content hash is invalid")
    last_date: date | None = None
    last_currency = ""
    seen: set[tuple[date, str]] = set()
    previous_by_currency: dict[str, tuple[date, float]] = {}
    currencies_by_date: dict[date, set[str]] = {}
    for row in rows:
        market_date = row["market_date"]
        currency = row["currency"]
        value = row["eur_per_unit"]
        if currency not in SUPPORTED_CURRENCIES:
            raise EurConversionError(f"Unsupported currency in derived data: {currency!r}")
        if not math.isfinite(value) or value <= 0:
            raise EurConversionError(f"Invalid EUR valuation for {currency} on {market_date}")
        key = (market_date, currency)
        if key in seen:
            raise EurConversionError(f"Duplicate derived date/currency row: {key}")
        seen.add(key)
        currencies_by_date.setdefault(market_date, set()).add(currency)
        if last_date is not None and (market_date, CURRENCY_ORDER.index(currency)) < (
            last_date,
            CURRENCY_ORDER.index(last_currency),
        ):
            raise EurConversionError("Derived observations are not sorted by date and currency")
        last_date, last_currency = market_date, currency
        if row["available_at"] is not None or row["availability_known"]:
            raise EurConversionError("FRED-derived availability must remain explicitly unknown")
        if any(source_date != market_date for source_date in row["source_market_dates"]):
            raise EurConversionError(
                f"Source market dates are not synchronized for {currency} on {market_date}"
            )
        provenance_lengths = {
            len(row["source_series_ids"]),
            len(row["source_raw_file_ids"]),
            len(row["source_parquet_files"]),
            len(row["source_market_dates"]),
        }
        if len(provenance_lengths) != 1:
            raise EurConversionError(
                f"Source provenance is incomplete for {currency} on {market_date}"
            )
        if row["source_series_ids"] != list(SERIES_BY_CURRENCY[currency]):
            raise EurConversionError(f"Unexpected source series for {currency} on {market_date}")
        if currency == "EUR" and value != 1.0:
            raise EurConversionError("The EUR numeraire rate must remain exactly 1")
        prior = previous_by_currency.get(currency)
        if prior is None:
            if (
                row["simple_return"] is not None
                or row["log_return"] is not None
                or row["previous_market_date"] is not None
                or row["calendar_days_since_previous"] is not None
            ):
                raise EurConversionError("The first return for each currency must be missing")
        else:
            prior_date, prior_value = prior
            expected_days = (market_date - prior_date).days
            if row["previous_market_date"] != prior_date:
                raise EurConversionError("Derived previous_market_date does not match chronology")
            if row["calendar_days_since_previous"] != expected_days:
                raise EurConversionError("Derived return interval length is inconsistent")
            expected_simple = value / prior_value - 1.0
            expected_log = math.log(value / prior_value)
            if not math.isclose(
                row["simple_return"], expected_simple, rel_tol=1e-12, abs_tol=1e-15
            ):
                raise EurConversionError(
                    "Derived simple return does not match adjacent observations"
                )
            if not math.isclose(row["log_return"], expected_log, rel_tol=1e-12, abs_tol=1e-15):
                raise EurConversionError("Derived log return does not match adjacent observations")
            if currency == "EUR" and (row["simple_return"] != 0.0 or row["log_return"] != 0.0):
                raise EurConversionError("EUR returns must be zero after the first observation")
        previous_by_currency[currency] = market_date, value

    expected_currencies = set(CURRENCY_ORDER)
    dates = sorted({row["market_date"] for row in rows})
    if len(rows) != len(dates) * len(expected_currencies):
        raise EurConversionError("Derived Parquet does not have a complete eight-currency calendar")
    for market_date, actual in currencies_by_date.items():
        if actual != expected_currencies:
            raise EurConversionError(f"Incomplete currency set on {market_date}")
    return EurParquetSnapshot(
        rows=rows,
        transform_version=transform_version,
        schema_version=schema_version,
        path=parquet_path,
    )


def _render_chart(snapshot: EurParquetSnapshot, output_path: Path) -> None:
    dates = sorted({row["market_date"] for row in snapshot.rows})
    by_currency = {
        currency: {
            row["market_date"]: row["eur_per_unit"]
            for row in snapshot.rows
            if row["currency"] == currency
        }
        for currency in CURRENCY_ORDER
    }
    figure, axis = plt.subplots(figsize=(13, 7))
    for currency in CURRENCY_ORDER:
        values = by_currency[currency]
        initial = values[dates[0]]
        normalized = [values[market_date] / initial * 100.0 for market_date in dates]
        axis.plot(dates, normalized, linewidth=1.2, label=currency)
    axis.set_title("Normalized value of one currency unit in EUR (start = 100)")
    axis.set_xlabel("Market date")
    axis.set_ylabel("Index (EUR per unit, normalized)")
    axis.grid(True, alpha=0.25)
    axis.legend(ncol=4)
    axis.xaxis.set_major_locator(mdates.YearLocator(base=4))
    axis.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    figure.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=150)
    plt.close(figure)


def write_validation_report(
    result: EurConversionResult,
    snapshot: EurParquetSnapshot,
    *,
    report_path: str | Path,
    chart_path: str | Path,
) -> None:
    """Write conversion validation metrics and the normalized-index chart."""
    report_path = Path(report_path)
    chart_path = Path(chart_path)
    rows_by_currency = {
        currency: [row for row in snapshot.rows if row["currency"] == currency]
        for currency in CURRENCY_ORDER
    }
    lines = [
        "# EUR conversion validation report",
        "",
        (
            "This dataset is for historical research. FRED H.10 rates are reference values, "
            "not executable broker prices."
        ),
        "",
        (
            "All FRED publication availability is unknown; `available_at` is null and "
            "`availability_known` is false. No forward-fill or interpolation is used."
        ),
        "",
        "## Calendar and conversion",
        "",
        f"- Common period: {result.common_dates[0]} through {result.common_dates[-1]}",
        f"- Common market dates: {len(result.common_dates)}",
        (
            "- Currency observations: "
            f"{len(snapshot.rows)} ({len(result.common_dates)} dates × 8 currencies)"
        ),
        f"- Candidate dates excluded because at least one source rate was unavailable: {len(result.excluded_dates)}",
        f"- Excluded dates: {', '.join(day.isoformat() for day in result.excluded_dates) or 'none'}",
        (
            "- Returns are changes between consecutive common observations. "
            "Each row records its actual calendar-day interval; gaps are not described as "
            "one-calendar-day returns."
        ),
        "",
        "## Currency values and returns",
        "",
        "| Currency | First EUR/unit | Last EUR/unit | Min simple return | Max simple return | Anomalies |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    anomaly_count = 0
    for currency in CURRENCY_ORDER:
        currency_rows = rows_by_currency[currency]
        returns = [
            row["simple_return"] for row in currency_rows if row["simple_return"] is not None
        ]
        if any(not math.isfinite(value) for value in returns):
            anomaly_count += 1
        lines.append(
            f"| {currency} | {currency_rows[0]['eur_per_unit']:.12g} "
            f"| {currency_rows[-1]['eur_per_unit']:.12g} "
            f"| {min(returns):.8%} | {max(returns):.8%} | "
            f"{'1' if any(not math.isfinite(value) for value in returns) else '0'} |"
        )
    lines.extend(
        [
            "",
            (
                "No heuristic outlier threshold is applied; anomalies here are invalid or "
                "non-finite derived values, invalid returns, duplicates, or incomplete "
                "currency rows. Missing source dates are reported separately above."
            ),
            f"- Detected derived anomalies: {anomaly_count}",
            (
                "- EUR is the numeraire at exactly 1 EUR/EUR; its returns are 0 after the first "
                "observation."
            ),
            "",
            "## Source snapshots",
            "",
            f"- FRED Parquet schema version: {FRED_PARQUET_SCHEMA_VERSION}",
            f"- EUR transformation version: {TRANSFORM_VERSION}",
            f"- Derived dataset schema version: {snapshot.schema_version}",
            "",
            "| FRED series | Requested window | Valid source rows | Raw SHA-256 | Raw file id | Parquet file |",
            "| --- | --- | ---: | --- | --- | --- |",
        ]
    )
    for series_id in FRED_FX_SERIES:
        source = result.sources[series_id]
        lines.append(
            f"| {series_id} | {source.request_start}–{source.request_end} "
            f"| {len(source.parquet.observations)} | `{source.raw.payload_sha256}` "
            f"| `{source.raw.raw_file_id}` | `{source.parquet.path.name}` |"
        )
    lines.extend(
        [
            "",
            "## Normalized index chart",
            "",
            "![Normalized currency values in EUR](eur_conversion_index.png)",
            "",
            (
                "The chart indexes each EUR-per-unit series to 100 at its first common date. "
                "It is a relative visualization, not a performance or investability claim."
            ),
            "",
        ]
    )
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text("\n".join(lines), encoding="utf-8")
    _render_chart(snapshot, chart_path)


def run_local_conversion(
    *,
    processed_root: str | Path | None = None,
    raw_root: str | Path | None = None,
    derived_root: str | Path | None = None,
    reports_dir: str | Path | None = None,
) -> EurConversionResult:
    """Validate the local full histories, derive EUR rates, and persist report outputs."""
    processed_root = Path(processed_root or PROJECT_ROOT / "data" / "processed")
    raw_root = Path(raw_root or PROJECT_ROOT / "data" / "raw")
    derived_root = Path(derived_root or PROJECT_ROOT / "data" / "derived")
    reports_dir = Path(reports_dir or PROJECT_ROOT / "reports")
    sources = select_complete_snapshots(processed_root, raw_root)
    result = build_eur_conversion(sources)
    parquet_snapshot = write_eur_parquet(derived_root, result.rows)
    result = EurConversionResult(
        sources=result.sources,
        rows=result.rows,
        common_dates=result.common_dates,
        excluded_dates=result.excluded_dates,
        derived_path=parquet_snapshot.path,
    )
    write_validation_report(
        result,
        parquet_snapshot,
        report_path=reports_dir / "eur_conversion_validation.md",
        chart_path=reports_dir / "eur_conversion_index.png",
    )
    return result


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Build EUR-denominated rates from verified local FRED Parquet files."
    )
    parser.add_argument("--processed-root", type=Path, default=PROJECT_ROOT / "data" / "processed")
    parser.add_argument("--raw-root", type=Path, default=PROJECT_ROOT / "data" / "raw")
    parser.add_argument("--derived-root", type=Path, default=PROJECT_ROOT / "data" / "derived")
    parser.add_argument("--reports-dir", type=Path, default=PROJECT_ROOT / "reports")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    result = run_local_conversion(
        processed_root=args.processed_root,
        raw_root=args.raw_root,
        derived_root=args.derived_root,
        reports_dir=args.reports_dir,
    )
    print(f"Derived Parquet: {result.derived_path}")
    print(
        f"Common period: {result.common_dates[0]} through {result.common_dates[-1]} "
        f"({len(result.common_dates)} dates; {len(result.rows)} rows)"
    )
    print(f"Excluded source dates: {len(result.excluded_dates)}")
    print(f"Report: {args.reports_dir / 'eur_conversion_validation.md'}")
    print(f"Chart: {args.reports_dir / 'eur_conversion_index.png'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
