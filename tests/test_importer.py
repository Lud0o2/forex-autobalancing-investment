import json
from datetime import UTC, date, datetime, timedelta

import pytest
import requests

from forex_v1.data.cli import main
from forex_v1.data.coverage import build_coverage_report, write_coverage_report
from forex_v1.data.importer import import_series_batch
from forex_v1.data.providers.fred import (
    FRED_FX_SERIES,
    FredClient,
    FredNetworkError,
    FredRawResponse,
)
from forex_v1.data.storage.parquet_store import read_parquet
from forex_v1.data.storage.raw_store import read_raw_response


class FakeSession:
    def __init__(self, outcomes):
        self.outcomes = list(outcomes)
        self.calls = []

    def get(self, url, *, params, timeout):
        self.calls.append({"url": url, "params": dict(params), "timeout": timeout})
        outcome = self.outcomes.pop(0)
        if isinstance(outcome, BaseException):
            raise outcome
        return outcome


class FakeResponse:
    status_code = 200

    def __init__(self, payload):
        self.payload = payload

    def json(self):
        return self.payload


class FakeFredClient:
    def __init__(self, failing_series=(), payload_overrides=None):
        self.failing_series = set(failing_series)
        self.payload_overrides = payload_overrides or {}
        self.calls = []

    def fetch_raw_series(self, series_id, *, start_date, end_date):
        self.calls.append((series_id, start_date, end_date))
        if series_id in self.failing_series:
            raise FredNetworkError(f"Network error while requesting FRED series {series_id}")
        payload = self.payload_overrides.get(
            series_id,
            {
                "count": "3",
                "offset": 0,
                "limit": 100000,
                "realtime_start": "2025-01-01",
                "realtime_end": "2025-01-01",
                "observations": [
                    {"date": "2025-01-02", "value": "1.08"},
                    {"date": "2025-01-03", "value": "."},
                    {"date": "2025-01-06", "value": "1.09"},
                ],
            },
        )
        return FredRawResponse(
            series_id=series_id,
            payload=payload,
            request_params={
                "series_id": series_id,
                "file_type": "json",
                "limit": 100000,
                "offset": 0,
                "observation_start": start_date.isoformat(),
                "observation_end": end_date.isoformat(),
            },
            retrieved_at=datetime(2025, 1, 7, tzinfo=UTC) + timedelta(seconds=len(self.calls)),
        )


@pytest.fixture(autouse=True)
def prohibit_unmocked_http(monkeypatch):
    def fail_request(*args, **kwargs):
        raise AssertionError("Importer tests must not issue network requests")

    monkeypatch.setattr(requests.sessions.Session, "request", fail_request)


def run_batch(client, root, series_ids=("DEXUSEU",)):
    return import_series_batch(
        client,
        series_ids,
        start_date=date(2025, 1, 1),
        end_date=date(2025, 1, 31),
        raw_root=root / "data" / "raw",
        processed_root=root / "data" / "processed",
    )


def test_import_saves_raw_and_reconstructs_parquet_idempotently(tmp_path):
    client = FakeFredClient()

    first = run_batch(client, tmp_path)[0]
    second = run_batch(client, tmp_path)[0]

    assert first.status == second.status == "success"
    assert first.raw_snapshot.payload["observations"][1]["value"] == "."
    assert first.raw_snapshot.payload_sha256 == second.raw_snapshot.payload_sha256
    assert first.raw_snapshot.raw_file_id != second.raw_snapshot.raw_file_id
    assert first.parquet_snapshot.path == second.parquet_snapshot.path
    assert len(first.parquet_snapshot.observations) == 2
    assert all(
        row.available_at is None and not row.availability_known
        for row in first.parquet_snapshot.observations
    )
    assert (
        read_raw_response(first.raw_snapshot.manifest_path).payload_sha256
        == first.raw_snapshot.payload_sha256
    )
    assert (
        read_parquet(first.parquet_snapshot.path).observations
        == first.parquet_snapshot.observations
    )
    assert len(list((tmp_path / "data" / "raw").rglob("*.manifest.json"))) == 2
    assert len(list((tmp_path / "data" / "processed").rglob("*.parquet"))) == 1
    assert first.raw_snapshot.request_params["offset"] == 0
    assert client.calls == [
        ("DEXUSEU", date(2025, 1, 1), date(2025, 1, 31)),
        ("DEXUSEU", date(2025, 1, 1), date(2025, 1, 31)),
    ]


def test_saved_import_files_do_not_contain_environment_api_key(monkeypatch, tmp_path):
    monkeypatch.setenv("FRED_API_KEY", "test-secret-never-store")
    result = run_batch(FakeFredClient(), tmp_path)[0]

    stored_files = [
        *result.raw_snapshot.response_path.parent.glob("*.json"),
        result.raw_snapshot.manifest_path,
    ]
    assert all(
        "test-secret-never-store" not in path.read_text(encoding="utf-8") for path in stored_files
    )


def test_partial_response_is_preserved_but_not_successfully_processed(tmp_path):
    client = FakeFredClient(
        payload_overrides={
            "DEXUSEU": {
                "count": "2",
                "offset": 0,
                "observations": [{"date": "2025-01-02", "value": "1.08"}],
            }
        }
    )

    result = run_batch(client, tmp_path)[0]

    assert result.status == "error"
    assert "incomplete" in result.error
    assert result.raw_snapshot is not None
    assert result.parquet_snapshot is None
    assert len(list((tmp_path / "data" / "raw").rglob("*.manifest.json"))) == 1


def test_out_of_range_observation_is_preserved_but_rejected(tmp_path):
    client = FakeFredClient(
        payload_overrides={
            "DEXUSEU": {
                "count": "1",
                "offset": 0,
                "observations": [{"date": "2024-12-31", "value": "1.08"}],
            }
        }
    )

    result = run_batch(client, tmp_path)[0]

    assert result.status == "error"
    assert "before the requested start date" in result.error
    assert result.raw_snapshot is not None
    assert result.parquet_snapshot is None


def test_invalid_price_is_reported_as_anomaly_and_raw_is_retained(tmp_path):
    client = FakeFredClient(
        payload_overrides={
            "DEXUSEU": {
                "count": "1",
                "offset": 0,
                "observations": [{"date": "2025-01-02", "value": "0"}],
            }
        }
    )

    result = run_batch(client, tmp_path)[0]
    report = build_coverage_report((result,))
    row = next(item for item in report["series"] if item["series_id"] == "DEXUSEU")

    assert result.status == "error"
    assert result.raw_snapshot is not None
    assert row["anomalous_prices"] == 1


def test_cli_execute_without_key_exits_without_network(monkeypatch, tmp_path):
    monkeypatch.delenv("FRED_API_KEY", raising=False)

    result = main(
        [
            "--execute",
            "--raw-root",
            str(tmp_path / "raw"),
            "--processed-root",
            str(tmp_path / "processed"),
            "--reports-dir",
            str(tmp_path / "reports"),
        ]
    )

    assert result == 2
    assert not (tmp_path / "raw").exists()
    assert not (tmp_path / "processed").exists()
    assert (tmp_path / "reports" / "data_coverage.json").exists()


def test_batch_reports_network_failure_for_one_series_and_continues(tmp_path):
    client = FakeFredClient(failing_series={"DEXUSUK"})

    results = run_batch(client, tmp_path, ("DEXUSEU", "DEXUSUK", "DEXJPUS"))

    assert [result.status for result in results] == ["success", "error", "success"]
    assert [call[0] for call in client.calls] == ["DEXUSEU", "DEXUSUK", "DEXJPUS"]


def test_missing_api_key_is_reported_without_network_or_secret(monkeypatch, tmp_path):
    monkeypatch.delenv("FRED_API_KEY", raising=False)
    client = FredClient(session=FakeSession([]), max_attempts=1)

    results = import_series_batch(
        client,
        ("DEXUSEU",),
        start_date=date(2025, 1, 1),
        end_date=date(2025, 1, 31),
        raw_root=tmp_path / "raw",
        processed_root=tmp_path / "processed",
    )

    assert results[0].status == "error"
    assert "FRED_API_KEY" in results[0].error
    assert client._session.calls == []


def test_network_failure_is_reported_without_persisting_an_invalid_response(monkeypatch, tmp_path):
    monkeypatch.setenv("FRED_API_KEY", "do-not-print-this")
    session = FakeSession([requests.ConnectionError()])
    client = FredClient(session=session, max_attempts=1, retry_backoff_seconds=0)

    result = run_batch(client, tmp_path)[0]

    assert result.status == "error"
    assert "FredNetworkError" in result.error
    assert "do-not-print-this" not in result.error
    assert result.raw_snapshot is None
    assert len(session.calls) == 1


def test_coverage_report_describes_all_seven_series_and_common_dates(tmp_path):
    client = FakeFredClient()
    results = run_batch(client, tmp_path, tuple(FRED_FX_SERIES))

    report = build_coverage_report(results)
    rows = {row["series_id"]: row for row in report["series"]}

    assert len(rows) == 7
    assert rows["DEXUSEU"]["valid_observation_count"] == 2
    assert rows["DEXUSEU"]["explicit_missing_value_count"] == 1
    assert rows["DEXUSEU"]["coverage_percent"] == 66.67
    assert rows["DEXUSEU"]["availability_known"] is False
    assert report["common_history"]["first_common_date"] == "2025-01-02"
    assert report["common_history"]["last_common_date"] == "2025-01-06"
    assert report["common_history"]["common_observation_count"] == 2
    assert report["point_in_time_certified"] is False

    markdown_path = tmp_path / "reports" / "data_coverage.md"
    json_path = tmp_path / "reports" / "data_coverage.json"
    written = write_coverage_report(
        results,
        markdown_path=markdown_path,
        json_path=json_path,
    )
    saved = json.loads(json_path.read_text(encoding="utf-8"))
    assert saved == written
    assert "not a point-in-time" in markdown_path.read_text(encoding="utf-8")


def test_dry_run_writes_report_without_network_or_data_files(tmp_path):
    result = main(
        [
            "--dry-run",
            "--raw-root",
            str(tmp_path / "raw"),
            "--processed-root",
            str(tmp_path / "processed"),
            "--reports-dir",
            str(tmp_path / "reports"),
        ]
    )

    assert result == 0
    assert not (tmp_path / "raw").exists()
    assert not (tmp_path / "processed").exists()
    assert (tmp_path / "reports" / "data_coverage.json").exists()
