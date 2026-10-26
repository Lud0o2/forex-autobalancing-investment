"""Orchestrate explicit FRED retrieval, raw preservation, and Parquet reconstruction."""

import os
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Any

from forex_v1.data.providers.fred import (
    FRED_FX_SERIES,
    FredClient,
    FredError,
    FredRawResponse,
    validate_series_id,
)
from forex_v1.data.storage.parquet_store import (
    ParquetSnapshot,
    ParquetStoreError,
    rebuild_from_raw,
)
from forex_v1.data.storage.raw_store import (
    RawFredSnapshot,
    RawStoreError,
    write_raw_response,
)


@dataclass(frozen=True, slots=True)
class SeriesImportResult:
    series_id: str
    status: str
    start_date: date
    end_date: date
    raw_snapshot: RawFredSnapshot | None = None
    parquet_snapshot: ParquetSnapshot | None = None
    error: str | None = None


def _safe_error(error: Exception) -> str:
    message = str(error)
    api_key = os.environ.get("FRED_API_KEY")
    if api_key:
        message = message.replace(api_key, "[REDACTED]")
    return f"{type(error).__name__}: {message}"


def _validate_complete_payload(response: FredRawResponse) -> None:
    observations = response.payload.get("observations")
    count = response.payload.get("count")
    if not isinstance(observations, list):
        raise TypeError("FRED response is missing its observations list")
    if isinstance(count, bool):
        raise TypeError("FRED response has an invalid count")
    if not isinstance(count, (str, int)) or isinstance(count, bool):
        raise TypeError("FRED response is missing a valid count")
    try:
        expected_count = int(count)
    except ValueError as error:
        raise ValueError("FRED response is missing a valid count") from error
    if expected_count < 0:
        raise ValueError("FRED response has a negative count")
    if expected_count != len(observations):
        raise ValueError(
            f"FRED response is incomplete: received {len(observations)} of {expected_count} "
            "observations"
        )
    offset = response.payload.get("offset", 0)
    if not isinstance(offset, (str, int)) or isinstance(offset, bool):
        raise TypeError("FRED response has an invalid offset")
    try:
        parsed_offset = int(offset)
    except ValueError as error:
        raise ValueError("FRED response has an invalid offset") from error
    if parsed_offset != 0:
        raise ValueError("FRED response is incomplete: first-page offset is not zero")


def import_series(
    client: Any,
    series_id: str,
    *,
    start_date: date,
    end_date: date,
    raw_root: str | Path,
    processed_root: str | Path,
) -> SeriesImportResult:
    """Import one series while retaining raw responses even if conversion later fails."""
    validate_series_id(series_id)
    if start_date > end_date:
        raise ValueError("start_date must not be after end_date")

    raw_snapshot = None
    parquet_snapshot = None
    try:
        response = client.fetch_raw_series(
            series_id,
            start_date=start_date,
            end_date=end_date,
        )
        if response.series_id != series_id:
            raise ValueError("FRED response series_id does not match the requested series")
        raw_snapshot = write_raw_response(raw_root, response)
        _validate_complete_payload(response)
        parquet_snapshot = rebuild_from_raw(raw_snapshot.manifest_path, processed_root)
        return SeriesImportResult(
            series_id=series_id,
            status="success",
            start_date=start_date,
            end_date=end_date,
            raw_snapshot=raw_snapshot,
            parquet_snapshot=parquet_snapshot,
        )
    except (
        FredError,
        RawStoreError,
        ParquetStoreError,
        TypeError,
        ValueError,
        OSError,
    ) as error:
        return SeriesImportResult(
            series_id=series_id,
            status="error",
            start_date=start_date,
            end_date=end_date,
            raw_snapshot=raw_snapshot,
            parquet_snapshot=parquet_snapshot,
            error=_safe_error(error),
        )


def import_series_batch(
    client: FredClient,
    series_ids: tuple[str, ...],
    *,
    start_date: date,
    end_date: date,
    raw_root: str | Path,
    processed_root: str | Path,
) -> tuple[SeriesImportResult, ...]:
    """Attempt each requested series independently so one provider failure is reported."""
    if not series_ids:
        raise ValueError("At least one FRED series must be selected")
    if len(set(series_ids)) != len(series_ids):
        raise ValueError("FRED series selections must not contain duplicates")
    for series_id in series_ids:
        validate_series_id(series_id)

    return tuple(
        import_series(
            client,
            series_id,
            start_date=start_date,
            end_date=end_date,
            raw_root=raw_root,
            processed_root=processed_root,
        )
        for series_id in series_ids
    )


def validate_selected_series(series_ids: tuple[str, ...]) -> None:
    """Validate CLI series selections."""
    if not series_ids:
        raise ValueError("At least one FRED series must be selected")
    for series_id in series_ids:
        validate_series_id(series_id)


ALL_FRED_SERIES = tuple(FRED_FX_SERIES)
