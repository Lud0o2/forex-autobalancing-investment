"""Versioned, immutable Parquet storage for standardized FRED FX observations."""

import hashlib
import json
import tempfile
from collections.abc import Iterable
from dataclasses import dataclass
from datetime import UTC, date, datetime
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

from forex_v1.data.providers.fred import FRED_FX_SERIES, parse_fred_observations
from forex_v1.data.schemas import FXObservation
from forex_v1.data.storage._files import ImmutableFileError, write_immutable_file
from forex_v1.data.storage.raw_store import (
    RawFredSnapshot,
    RawStoreError,
    read_raw_response,
)
from forex_v1.data.validation import validate_observations

PARQUET_SCHEMA_VERSION = 1
PARQUET_SCHEMA = pa.schema(
    [
        pa.field("market_date", pa.date32(), nullable=False),
        pa.field("base_currency", pa.string(), nullable=False),
        pa.field("quote_currency", pa.string(), nullable=False),
        pa.field("quote_per_base", pa.float64(), nullable=False),
        pa.field("source", pa.string(), nullable=False),
        pa.field("series_id", pa.string(), nullable=False),
        pa.field("price_type", pa.string(), nullable=False),
        pa.field("available_at", pa.timestamp("us", tz="UTC"), nullable=True),
        pa.field("availability_known", pa.bool_(), nullable=False),
        pa.field("raw_file_id", pa.string(), nullable=False),
        pa.field("schema_version", pa.int16(), nullable=False),
    ]
)
SCHEMA_VERSION_KEY = b"forex_v1.schema_version"
SERIES_ID_KEY = b"forex_v1.series_id"
CONTENT_SHA256_KEY = b"forex_v1.content_sha256"


class ParquetStoreError(RuntimeError):
    """Raised when processed observations cannot be safely written or read."""


@dataclass(frozen=True, slots=True)
class ParquetSnapshot:
    """Validated rows and provenance loaded from a versioned Parquet file."""

    series_id: str
    observations: tuple[FXObservation, ...]
    raw_file_ids: tuple[str, ...]
    schema_version: int
    path: Path


def _validate_for_series(
    series_id: str,
    observations: Iterable[FXObservation],
    raw_file_id: str,
) -> tuple[FXObservation, ...]:
    if series_id not in FRED_FX_SERIES:
        raise ParquetStoreError(f"Unsupported FRED series: {series_id!r}")
    if not isinstance(raw_file_id, str) or not raw_file_id.strip():
        raise ParquetStoreError("raw_file_id must be a non-empty path relative to data/raw")

    rows = tuple(observations)
    if not rows:
        raise ParquetStoreError("Refusing to write an empty processed dataset")
    try:
        validate_observations(rows)
    except (TypeError, ValueError) as error:
        raise ParquetStoreError(f"Invalid observations: {error}") from error

    expected_pair = FRED_FX_SERIES[series_id]
    for observation in rows:
        if (observation.base_currency, observation.quote_currency) != expected_pair:
            raise ParquetStoreError(
                f"Observation pair does not match FRED series {series_id}"
            )
        if observation.source != f"fred_h10:{series_id}":
            raise ParquetStoreError(
                f"Observation source does not identify FRED series {series_id}"
            )
    return tuple(sorted(rows, key=lambda observation: observation.market_date))


def _record(
    observation: FXObservation,
    series_id: str,
    raw_file_id: str,
) -> dict[str, object]:
    available_at = observation.available_at
    if available_at is not None:
        available_at = available_at.astimezone(UTC)
    return {
        "market_date": observation.market_date,
        "base_currency": observation.base_currency,
        "quote_currency": observation.quote_currency,
        "quote_per_base": observation.quote_per_base,
        "source": observation.source,
        "series_id": series_id,
        "price_type": observation.price_type.value,
        "available_at": available_at,
        "availability_known": observation.availability_known,
        "raw_file_id": raw_file_id,
        "schema_version": PARQUET_SCHEMA_VERSION,
    }


def _canonical_rows(records: list[dict[str, object]]) -> bytes:
    serializable = []
    for record in records:
        row = dict(record)
        market_date = row["market_date"]
        available_at = row["available_at"]
        row["market_date"] = market_date.isoformat() if isinstance(market_date, date) else market_date
        row["available_at"] = (
            available_at.isoformat() if isinstance(available_at, datetime) else None
        )
        serializable.append(row)
    return json.dumps(
        serializable,
        allow_nan=False,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")


def write_observations(
    processed_root: str | Path,
    series_id: str,
    observations: Iterable[FXObservation],
    *,
    raw_file_id: str,
) -> ParquetSnapshot:
    """Validate and atomically write an immutable, content-addressed Parquet snapshot."""
    rows = _validate_for_series(series_id, observations, raw_file_id)
    series_directory = Path(processed_root) / "fred" / series_id
    if series_directory.exists():
        for existing_path in sorted(series_directory.glob("*.parquet")):
            existing = read_parquet(existing_path)
            if existing.observations == rows:
                return existing

    records = [_record(observation, series_id, raw_file_id) for observation in rows]
    content_digest = hashlib.sha256(_canonical_rows(records)).hexdigest()
    output_path = (
        Path(processed_root)
        / "fred"
        / series_id
        / f"{series_id}_{content_digest}.parquet"
    )
    metadata = {
        SCHEMA_VERSION_KEY: str(PARQUET_SCHEMA_VERSION).encode("ascii"),
        SERIES_ID_KEY: series_id.encode("ascii"),
        CONTENT_SHA256_KEY: content_digest.encode("ascii"),
    }
    table = pa.Table.from_pylist(records, schema=PARQUET_SCHEMA.with_metadata(metadata))
    if output_path.exists():
        existing = read_parquet(output_path)
        if (
            existing.series_id != series_id
            or existing.observations != rows
            or existing.raw_file_ids != (raw_file_id,) * len(rows)
        ):
            raise ParquetStoreError(f"Existing content-addressed Parquet file differs: {output_path}")
        return existing

    try:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        temporary_path: Path | None = None
        try:
            with tempfile.NamedTemporaryFile(
                prefix=f".{output_path.name}.",
                suffix=".tmp",
                dir=output_path.parent,
                delete=False,
            ) as temporary_file:
                temporary_path = Path(temporary_file.name)
            pq.write_table(table, temporary_path, compression="zstd", version="2.6")
            write_immutable_file(output_path, temporary_path.read_bytes())
        finally:
            if temporary_path is not None:
                temporary_path.unlink(missing_ok=True)
    except (OSError, pa.ArrowException, ImmutableFileError, FileExistsError) as error:
        if isinstance(error, FileExistsError) and output_path.exists():
            existing = read_parquet(output_path)
            if (
                existing.series_id == series_id
                and existing.observations == rows
                and existing.raw_file_ids == (raw_file_id,) * len(rows)
            ):
                return existing
        raise ParquetStoreError(f"Could not safely write Parquet data: {error}") from error

    snapshot = read_parquet(output_path)
    if snapshot.observations != rows or snapshot.raw_file_ids != (raw_file_id,) * len(rows):
        raise ParquetStoreError("Written Parquet data does not match the requested observations")
    return snapshot


def read_parquet(path: str | Path) -> ParquetSnapshot:
    """Load a Parquet snapshot, enforce its schema version, and validate its observations."""
    parquet_path = Path(path)
    try:
        table = pq.read_table(parquet_path)
    except (OSError, pa.ArrowException) as error:
        raise ParquetStoreError(f"Could not read Parquet file {parquet_path}: {error}") from error

    metadata = table.schema.metadata or {}
    try:
        schema_version = int(metadata[SCHEMA_VERSION_KEY].decode("ascii"))
        series_id = metadata[SERIES_ID_KEY].decode("ascii")
        expected_content_digest = metadata[CONTENT_SHA256_KEY].decode("ascii")
    except (KeyError, UnicodeDecodeError, ValueError) as error:
        raise ParquetStoreError("Parquet file is missing valid Forex schema metadata") from error
    if schema_version != PARQUET_SCHEMA_VERSION:
        raise ParquetStoreError(f"Unsupported Parquet schema version: {schema_version}")
    if series_id not in FRED_FX_SERIES:
        raise ParquetStoreError(f"Parquet file declares an unsupported FRED series: {series_id}")
    if not table.schema.remove_metadata().equals(PARQUET_SCHEMA):
        raise ParquetStoreError("Parquet file does not match the expected Forex observation schema")

    try:
        records = table.to_pylist()
        actual_content_digest = hashlib.sha256(_canonical_rows(records)).hexdigest()
        if actual_content_digest != expected_content_digest:
            raise ParquetStoreError("Parquet row content hash does not match its metadata")
        observations = tuple(
            FXObservation(
                market_date=row["market_date"],
                base_currency=row["base_currency"],
                quote_currency=row["quote_currency"],
                quote_per_base=row["quote_per_base"],
                source=row["source"],
                price_type=row["price_type"],
                available_at=row["available_at"],
                availability_known=row["availability_known"],
            )
            for row in records
        )
        raw_file_ids = tuple(row["raw_file_id"] for row in records)
    except (KeyError, TypeError, ValueError) as error:
        raise ParquetStoreError(f"Parquet rows violate the Forex observation contract: {error}") from error

    if not records or any(row["schema_version"] != schema_version for row in records):
        raise ParquetStoreError("Parquet rows are empty or have inconsistent schema versions")
    if len(set(raw_file_ids)) != 1 or any(not isinstance(raw_id, str) or not raw_id for raw_id in raw_file_ids):
        raise ParquetStoreError("Parquet rows must reference one non-empty raw source file")
    if [observation.market_date for observation in observations] != sorted(
        observation.market_date for observation in observations
    ):
        raise ParquetStoreError("Parquet observations are not sorted chronologically")

    validated = _validate_for_series(series_id, observations, raw_file_ids[0])
    if validated != observations:
        raise ParquetStoreError("Parquet observations are not in canonical chronological order")
    return ParquetSnapshot(
        series_id=series_id,
        observations=observations,
        raw_file_ids=raw_file_ids,
        schema_version=schema_version,
        path=parquet_path,
    )


def rebuild_from_raw(
    manifest_path: str | Path,
    processed_root: str | Path,
) -> ParquetSnapshot:
    """Rebuild one processed Parquet snapshot from a verified raw FRED response."""
    try:
        raw_snapshot: RawFredSnapshot = read_raw_response(manifest_path)
    except RawStoreError as error:
        raise ParquetStoreError(f"Cannot rebuild from raw response: {error}") from error
    request_params = raw_snapshot.request_params
    requested_dates: dict[str, date | None] = {}
    for parameter, field_name in (
        ("observation_start", "start_date"),
        ("observation_end", "end_date"),
    ):
        raw_date = request_params.get(parameter)
        if raw_date is None:
            requested_dates[field_name] = None
            continue
        if not isinstance(raw_date, str):
            raise ParquetStoreError(f"Raw request parameter {parameter} must be a date string")
        try:
            requested_dates[field_name] = date.fromisoformat(raw_date)
        except ValueError as error:
            raise ParquetStoreError(
                f"Raw request parameter {parameter} is not a valid ISO date"
            ) from error
    observations = parse_fred_observations(
        raw_snapshot.series_id,
        raw_snapshot.payload,
        start_date=requested_dates["start_date"],
        end_date=requested_dates["end_date"],
    )
    return write_observations(
        processed_root,
        raw_snapshot.series_id,
        observations,
        raw_file_id=raw_snapshot.raw_file_id,
    )
