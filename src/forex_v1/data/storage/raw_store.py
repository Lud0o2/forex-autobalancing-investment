"""Immutable storage and integrity checks for complete FRED JSON responses."""

import hashlib
import json
import os
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from forex_v1.data.providers.fred import FRED_API_URL, FRED_FX_SERIES, FredRawResponse
from forex_v1.data.storage._files import ImmutableFileError, write_immutable_file

RAW_MANIFEST_VERSION = 1
SENSITIVE_KEY_PARTS = ("api_key", "authorization", "password", "secret", "token")
SAFE_REQUEST_PARAMS = frozenset(
    {
        "file_type",
        "limit",
        "observation_end",
        "observation_start",
        "offset",
        "series_id",
        "sort_order",
    }
)
FRED_METADATA_FIELDS = (
    "count",
    "offset",
    "limit",
    "realtime_start",
    "realtime_end",
    "observation_start",
    "observation_end",
)


class RawStoreError(RuntimeError):
    """Raised when a raw response cannot be safely stored or verified."""


@dataclass(frozen=True, slots=True)
class RawFredSnapshot:
    """Verified raw response and manifest information."""

    series_id: str
    payload: Mapping[str, Any]
    request_params: Mapping[str, str | int]
    retrieved_at: datetime
    payload_sha256: str
    raw_file_id: str
    response_path: Path
    manifest_path: Path
    fred_metadata: Mapping[str, Any]


def _canonical_json(value: Any) -> bytes:
    try:
        return json.dumps(
            value,
            ensure_ascii=False,
            allow_nan=False,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("utf-8")
    except (TypeError, ValueError) as error:
        raise RawStoreError("FRED response contains values that cannot be safely serialized") from error


def _reject_sensitive_keys(value: Any, location: str = "response") -> None:
    if isinstance(value, Mapping):
        for key, nested in value.items():
            normalized = str(key).lower().replace("-", "_")
            if any(part in normalized for part in SENSITIVE_KEY_PARTS):
                raise RawStoreError(f"Sensitive field is not allowed in stored {location}")
            _reject_sensitive_keys(nested, location)
    elif isinstance(value, list):
        for nested in value:
            _reject_sensitive_keys(nested, location)


def write_raw_response(
    raw_root: str | Path,
    response: FredRawResponse,
    *,
    api_key: str | None = None,
) -> RawFredSnapshot:
    """Write a complete response and manifest without storing credentials or overwriting data."""
    if response.series_id not in FRED_FX_SERIES:
        raise RawStoreError(f"Unsupported FRED series: {response.series_id!r}")
    if response.retrieved_at.tzinfo is None or response.retrieved_at.utcoffset() is None:
        raise RawStoreError("retrieved_at must be a timezone-aware timestamp")
    retrieved_at = response.retrieved_at.astimezone(UTC)
    _reject_sensitive_keys(response.payload)
    _reject_sensitive_keys(response.request_params, "request parameters")

    if set(response.request_params) - SAFE_REQUEST_PARAMS:
        raise RawStoreError("Request parameters contain unsupported or sensitive fields")
    if response.request_params.get("series_id") != response.series_id:
        raise RawStoreError("Request series_id does not match the FRED response")
    key_value = api_key or os.environ.get("FRED_API_KEY")
    if not isinstance(response.payload, Mapping) or not isinstance(
        response.payload.get("observations"), list
    ):
        raise RawStoreError("FRED response must contain an observations list")

    body = _canonical_json(response.payload)
    if key_value and key_value.encode("utf-8") in body:
        raise RawStoreError("FRED response unexpectedly contains the API key")
    payload_sha256 = hashlib.sha256(body).hexdigest()
    timestamp = retrieved_at.strftime("%Y%m%dT%H%M%S%fZ")
    basename = f"{response.series_id}_{timestamp}_{payload_sha256}"
    response_relative_path = Path("fred") / response.series_id / f"{basename}.json"
    response_path = Path(raw_root) / response_relative_path
    manifest_path = response_path.with_suffix(".manifest.json")
    fred_metadata = {
        key: response.payload[key]
        for key in FRED_METADATA_FIELDS
        if key in response.payload
    }
    manifest = {
        "manifest_version": RAW_MANIFEST_VERSION,
        "provider": "fred",
        "endpoint": FRED_API_URL,
        "series_id": response.series_id,
        "raw_file_id": response_relative_path.as_posix(),
        "retrieved_at": retrieved_at.isoformat(),
        "request_params": dict(response.request_params),
        "fred_metadata": fred_metadata,
        "payload_sha256": payload_sha256,
        "payload_bytes": len(body),
    }
    _reject_sensitive_keys(manifest)
    manifest_bytes = _canonical_json(manifest)
    if key_value and key_value.encode("utf-8") in manifest_bytes:
        raise RawStoreError("FRED manifest unexpectedly contains the API key")

    try:
        write_immutable_file(response_path, body)
        write_immutable_file(manifest_path, manifest_bytes)
    except (ImmutableFileError, FileExistsError) as error:
        raise RawStoreError(f"Could not safely store raw FRED response: {error}") from error
    return RawFredSnapshot(
        series_id=response.series_id,
        payload=response.payload,
        request_params=response.request_params,
        retrieved_at=retrieved_at,
        payload_sha256=payload_sha256,
        raw_file_id=response_relative_path.as_posix(),
        response_path=response_path,
        manifest_path=manifest_path,
        fred_metadata=fred_metadata,
    )


def read_raw_response(manifest_path: str | Path) -> RawFredSnapshot:
    """Load a raw response using its manifest and verify the SHA-256 digest."""
    path = Path(manifest_path)
    try:
        manifest_bytes = path.read_bytes()
        manifest = json.loads(manifest_bytes)
    except OSError as error:
        raise RawStoreError(f"Could not read raw manifest {path}: {error}") from error
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise RawStoreError(f"Raw manifest is corrupted: {path}") from error

    if not isinstance(manifest, Mapping):
        raise RawStoreError(f"Raw manifest is invalid: {path}")
    _reject_sensitive_keys(manifest, "manifest")
    if manifest.get("manifest_version") != RAW_MANIFEST_VERSION:
        raise RawStoreError("Unsupported raw manifest version")
    if manifest.get("provider") != "fred" or manifest.get("endpoint") != FRED_API_URL:
        raise RawStoreError("Raw manifest provider or endpoint is invalid")
    series_id = manifest.get("series_id")
    if not isinstance(series_id, str) or series_id not in FRED_FX_SERIES:
        raise RawStoreError("Raw manifest has an unsupported FRED series")

    raw_file_id = manifest.get("raw_file_id")
    if not isinstance(raw_file_id, str):
        raise RawStoreError("Raw manifest is missing its raw_file_id")
    relative_path = Path(raw_file_id)
    if (
        relative_path.is_absolute()
        or ".." in relative_path.parts
        or len(relative_path.parts) != 3
        or relative_path.parts[0] != "fred"
        or relative_path.parts[1] != series_id
    ):
        raise RawStoreError("Raw manifest raw_file_id must be a safe relative path")
    response_path = path.parent.parent.parent / relative_path
    if response_path.name != Path(raw_file_id).name:
        raise RawStoreError("Raw manifest response path does not match its raw_file_id")

    expected_digest = manifest.get("payload_sha256")
    if (
        not isinstance(expected_digest, str)
        or len(expected_digest) != 64
        or any(character not in "0123456789abcdef" for character in expected_digest)
    ):
        raise RawStoreError("Raw manifest has an invalid SHA-256 digest")
    try:
        body = response_path.read_bytes()
    except OSError as error:
        raise RawStoreError(f"Could not read raw response {response_path}: {error}") from error
    actual_digest = hashlib.sha256(body).hexdigest()
    if actual_digest != expected_digest:
        raise RawStoreError(f"Raw response SHA-256 mismatch: {response_path}")

    try:
        payload = json.loads(body)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise RawStoreError(f"Raw response is corrupted: {response_path}") from error
    if not isinstance(payload, Mapping) or not isinstance(payload.get("observations"), list):
        raise RawStoreError(f"Raw response has an invalid observations structure: {response_path}")
    _reject_sensitive_keys(payload)

    request_params = manifest.get("request_params")
    if not isinstance(request_params, Mapping) or set(request_params) - SAFE_REQUEST_PARAMS:
        raise RawStoreError("Raw manifest request parameters are invalid")
    if request_params.get("series_id") != series_id:
        raise RawStoreError("Raw manifest request series does not match its response")
    retrieved_at_value = manifest.get("retrieved_at")
    if not isinstance(retrieved_at_value, str):
        raise RawStoreError("Raw manifest has no retrieval timestamp")
    try:
        retrieved_at = datetime.fromisoformat(retrieved_at_value)
    except ValueError as error:
        raise RawStoreError("Raw manifest has an invalid retrieval timestamp") from error
    if retrieved_at.tzinfo is None or retrieved_at.utcoffset() is None:
        raise RawStoreError("Raw manifest retrieval timestamp must include a timezone")
    if retrieved_at.utcoffset().total_seconds() != 0:
        raise RawStoreError("Raw manifest retrieval timestamp must be UTC")

    fred_metadata = manifest.get("fred_metadata")
    if not isinstance(fred_metadata, Mapping):
        raise RawStoreError("Raw manifest FRED metadata must be an object")
    expected_metadata = {
        key: payload[key]
        for key in FRED_METADATA_FIELDS
        if key in payload
    }
    if dict(fred_metadata) != expected_metadata:
        raise RawStoreError("Raw manifest FRED metadata does not match the response")
    if manifest.get("payload_bytes") != len(body):
        raise RawStoreError("Raw response size does not match its manifest")

    return RawFredSnapshot(
        series_id=series_id,
        payload=payload,
        request_params=request_params,
        retrieved_at=retrieved_at.astimezone(UTC),
        payload_sha256=actual_digest,
        raw_file_id=raw_file_id,
        response_path=response_path,
        manifest_path=path,
        fred_metadata=fred_metadata,
    )
