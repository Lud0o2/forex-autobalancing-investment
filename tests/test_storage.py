import hashlib
import json
from datetime import UTC, date, datetime

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from forex_v1.data.providers.fred import (
    FRED_FX_SERIES,
    FredRawResponse,
    parse_fred_observations,
)
from forex_v1.data.schemas import FXObservation, PriceType
from forex_v1.data.storage.parquet_store import (
    PARQUET_SCHEMA_VERSION,
    ParquetStoreError,
    read_parquet,
    rebuild_from_raw,
    write_observations,
)
from forex_v1.data.storage.raw_store import (
    RawStoreError,
    read_raw_response,
    write_raw_response,
)


def fixture_payload():
    return {
        "realtime_start": "2025-01-07",
        "realtime_end": "2025-01-07",
        "observation_start": "2025-01-02",
        "observation_end": "2025-01-06",
        "count": 3,
        "offset": 0,
        "limit": 100000,
        "observations": [
            {"date": "2025-01-02", "value": "1.08"},
            {"date": "2025-01-03", "value": "."},
            {"date": "2025-01-06", "value": "1.09"},
        ],
    }


def raw_response(payload=None, *, retrieved_at=None, request_params=None):
    return FredRawResponse(
        series_id="DEXUSEU",
        payload=payload or fixture_payload(),
        request_params=request_params
        or {
            "file_type": "json",
            "limit": 100000,
            "series_id": "DEXUSEU",
            "sort_order": "asc",
        },
        retrieved_at=retrieved_at or datetime(2025, 1, 8, 12, 30, tzinfo=UTC),
    )


def write_fixture_raw(tmp_path):
    return write_raw_response(tmp_path / "raw", raw_response())


def test_raw_response_and_manifest_round_trip_with_complete_payload(tmp_path):
    saved = write_fixture_raw(tmp_path)

    loaded = read_raw_response(saved.manifest_path)

    assert loaded.payload == fixture_payload()
    assert loaded.payload["observations"][1]["value"] == "."
    assert loaded.fred_metadata["count"] == 3
    assert loaded.fred_metadata["offset"] == 0
    assert loaded.fred_metadata["limit"] == 100000
    assert loaded.fred_metadata["realtime_start"] == "2025-01-07"
    assert loaded.request_params["series_id"] == "DEXUSEU"
    assert loaded.retrieved_at == datetime(2025, 1, 8, 12, 30, tzinfo=UTC)
    assert loaded.raw_file_id == saved.raw_file_id
    assert loaded.payload_sha256 == hashlib.sha256(
        json.dumps(
            fixture_payload(),
            ensure_ascii=False,
            allow_nan=False,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("utf-8")
    ).hexdigest()


def test_raw_storage_is_idempotent_for_identical_snapshot(tmp_path):
    first = write_fixture_raw(tmp_path)
    second = write_raw_response(tmp_path / "raw", raw_response())

    assert first.raw_file_id == second.raw_file_id
    assert first.manifest_path.read_bytes() == second.manifest_path.read_bytes()
    assert len(list((tmp_path / "raw").rglob("*.json"))) == 2


def test_raw_manifest_does_not_store_api_key_or_auth_fields(tmp_path):
    secret = "not-a-real-secret-key"
    saved = write_raw_response(tmp_path / "raw", raw_response(), api_key=secret)

    for path in (saved.response_path, saved.manifest_path):
        content = path.read_bytes()
        assert secret.encode() not in content
        assert b"api_key" not in content
        assert b"authorization" not in content.lower()


def test_raw_storage_rejects_sensitive_fields_and_query_parameters(tmp_path):
    payload = fixture_payload()
    payload["api_key"] = "leak"
    with pytest.raises(RawStoreError, match="Sensitive field"):
        write_raw_response(tmp_path / "raw", raw_response(payload))

    with pytest.raises(RawStoreError, match="Sensitive field"):
        write_raw_response(
            tmp_path / "raw",
            raw_response(request_params={"api_key": "leak", "series_id": "DEXUSEU"}),
        )


def test_raw_storage_refuses_to_overwrite_different_manifest(tmp_path):
    first = raw_response()
    write_raw_response(tmp_path / "raw", first)
    changed_params = dict(first.request_params)
    changed_params["observation_start"] = "2025-01-01"

    with pytest.raises(RawStoreError, match="safely store"):
        write_raw_response(
            tmp_path / "raw",
            raw_response(
                retrieved_at=first.retrieved_at,
                request_params=changed_params,
            ),
        )


def test_raw_corruption_is_detected_by_hash(tmp_path):
    saved = write_fixture_raw(tmp_path)
    saved.response_path.write_bytes(b'{"observations":[]}')

    with pytest.raises(RawStoreError, match="SHA-256 mismatch"):
        read_raw_response(saved.manifest_path)


def test_processed_observations_round_trip_and_preserve_unknown_availability(tmp_path):
    observations = parse_fred_observations("DEXUSEU", fixture_payload())
    snapshot = write_observations(
        tmp_path / "processed",
        "DEXUSEU",
        observations,
        raw_file_id="fred/DEXUSEU/source.json",
    )
    loaded = read_parquet(snapshot.path)

    assert snapshot.schema_version == PARQUET_SCHEMA_VERSION
    assert loaded.observations == observations
    assert loaded.raw_file_ids == ("fred/DEXUSEU/source.json",) * 2
    assert [row.market_date for row in loaded.observations] == [
        date(2025, 1, 2),
        date(2025, 1, 6),
    ]
    assert all(row.base_currency == "EUR" and row.quote_currency == "USD" for row in loaded.observations)
    assert all(row.price_type is PriceType.REFERENCE for row in loaded.observations)
    assert all(row.available_at is None and not row.availability_known for row in loaded.observations)


def test_processed_storage_preserves_timezone_aware_availability(tmp_path):
    observation = FXObservation(
        market_date=date(2025, 1, 2),
        base_currency="EUR",
        quote_currency="USD",
        quote_per_base=1.08,
        source="fred_h10:DEXUSEU",
        price_type=PriceType.REFERENCE,
        available_at=datetime.fromisoformat("2025-01-03T08:30:00+05:30"),
        availability_known=True,
    )
    snapshot = write_observations(
        tmp_path / "processed",
        "DEXUSEU",
        [observation],
        raw_file_id="fred/DEXUSEU/source.json",
    )

    loaded = read_parquet(snapshot.path).observations[0]

    assert loaded == observation
    assert loaded.available_at is not None
    assert loaded.available_at.utcoffset().total_seconds() == 0
    assert loaded.availability_known


def test_processed_storage_is_idempotent_for_same_data_and_raw_reference(tmp_path):
    observations = parse_fred_observations("DEXUSEU", fixture_payload())
    first = write_observations(
        tmp_path / "processed",
        "DEXUSEU",
        observations,
        raw_file_id="fred/DEXUSEU/source.json",
    )
    second = write_observations(
        tmp_path / "processed",
        "DEXUSEU",
        observations,
        raw_file_id="fred/DEXUSEU/source.json",
    )

    assert first.path == second.path
    assert len(list((tmp_path / "processed").rglob("*.parquet"))) == 1


def test_processed_file_retains_versions_when_observations_change(tmp_path):
    observations = parse_fred_observations("DEXUSEU", fixture_payload())
    corrected = (
        observations[0],
        FXObservation(
            market_date=observations[1].market_date,
            base_currency="EUR",
            quote_currency="USD",
            quote_per_base=1.10,
            source="fred_h10:DEXUSEU",
            price_type=PriceType.REFERENCE,
        ),
    )
    first = write_observations(
        tmp_path / "processed",
        "DEXUSEU",
        observations,
        raw_file_id="fred/DEXUSEU/first.json",
    )
    second = write_observations(
        tmp_path / "processed",
        "DEXUSEU",
        corrected,
        raw_file_id="fred/DEXUSEU/corrected.json",
    )

    assert first.path != second.path
    assert len(list((tmp_path / "processed").rglob("*.parquet"))) == 2
    assert read_parquet(first.path).observations == observations
    assert read_parquet(second.path).observations == corrected


def test_processed_duplicate_observations_are_rejected(tmp_path):
    observations = parse_fred_observations("DEXUSEU", fixture_payload())

    with pytest.raises(ParquetStoreError, match="Duplicate observation"):
        write_observations(
            tmp_path / "processed",
            "DEXUSEU",
            [observations[0], observations[0]],
            raw_file_id="fred/DEXUSEU/source.json",
        )


def test_processed_store_rejects_pair_mismatch(tmp_path):
    observations = parse_fred_observations("DEXUSEU", fixture_payload())

    with pytest.raises(ParquetStoreError, match="pair"):
        write_observations(
            tmp_path / "processed",
            "DEXJPUS",
            observations,
            raw_file_id="fred/DEXUSEU/source.json",
        )


def test_parquet_corruption_is_detected(tmp_path):
    observations = parse_fred_observations("DEXUSEU", fixture_payload())
    snapshot = write_observations(
        tmp_path / "processed",
        "DEXUSEU",
        observations,
        raw_file_id="fred/DEXUSEU/source.json",
    )
    snapshot.path.write_bytes(b"not parquet")

    with pytest.raises(ParquetStoreError, match="Could not read Parquet"):
        read_parquet(snapshot.path)


def test_parquet_row_tampering_is_detected_by_content_hash(tmp_path):
    observations = parse_fred_observations("DEXUSEU", fixture_payload())
    snapshot = write_observations(
        tmp_path / "processed",
        "DEXUSEU",
        observations,
        raw_file_id="fred/DEXUSEU/source.json",
    )
    table = pq.read_table(snapshot.path)
    rate_column = table.schema.get_field_index("quote_per_base")
    table = table.set_column(
        rate_column,
        table.schema.field(rate_column),
        pa.array([1.07, 1.09], type=pa.float64()),
    )
    pq.write_table(table, snapshot.path)

    with pytest.raises(ParquetStoreError, match="row content hash"):
        read_parquet(snapshot.path)


def test_processed_data_can_be_rebuilt_from_verified_raw_file(tmp_path):
    raw = write_fixture_raw(tmp_path)

    rebuilt = rebuild_from_raw(raw.manifest_path, tmp_path / "processed")

    assert rebuilt.series_id == "DEXUSEU"
    assert rebuilt.raw_file_ids == (raw.raw_file_id,) * 2
    assert [row.quote_per_base for row in rebuilt.observations] == [1.08, 1.09]
    assert all(not row.availability_known for row in rebuilt.observations)


def test_storage_supports_all_mapped_series():
    assert set(FRED_FX_SERIES) == {
        "DEXUSEU",
        "DEXUSUK",
        "DEXJPUS",
        "DEXSZUS",
        "DEXCAUS",
        "DEXUSAL",
        "DEXUSNZ",
    }
