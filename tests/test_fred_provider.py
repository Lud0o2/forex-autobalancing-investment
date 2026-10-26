from datetime import date, datetime

import pytest
import requests

from forex_v1.data.providers.fred import (
    FRED_API_URL,
    FRED_FX_SERIES,
    FRED_SOURCE,
    FredClient,
    FredConfigurationError,
    FredHTTPError,
    FredNetworkError,
    FredResponseError,
    parse_fred_observations,
)
from forex_v1.data.schemas import PriceType


class FakeResponse:
    def __init__(self, payload=None, status_code=200, json_error=None):
        self.payload = payload
        self.status_code = status_code
        self.json_error = json_error

    def json(self):
        if self.json_error is not None:
            raise self.json_error
        return self.payload


class FakeSession:
    def __init__(self, outcomes):
        self.outcomes = list(outcomes)
        self.calls = []

    def get(self, url, *, params, timeout):
        self.calls.append({"url": url, "params": params, "timeout": timeout})
        outcome = self.outcomes.pop(0)
        if isinstance(outcome, BaseException):
            raise outcome
        return outcome


@pytest.fixture(autouse=True)
def prohibit_unmocked_http(monkeypatch):
    def fail_request(*args, **kwargs):
        raise AssertionError("FRED tests must not issue network requests")

    monkeypatch.setattr(requests.sessions.Session, "request", fail_request)


def fred_payload(date_string="2025-01-02", value="1.08"):
    return {
        "count": "1",
        "observations": [{"date": date_string, "value": value}],
    }


@pytest.mark.parametrize(
    ("series_id", "base_currency", "quote_currency"),
    [
        ("DEXUSEU", "EUR", "USD"),
        ("DEXUSUK", "GBP", "USD"),
        ("DEXJPUS", "USD", "JPY"),
        ("DEXSZUS", "USD", "CHF"),
        ("DEXCAUS", "USD", "CAD"),
        ("DEXUSAL", "AUD", "USD"),
        ("DEXUSNZ", "NZD", "USD"),
    ],
)
def test_all_seven_series_convert_without_inversion(series_id, base_currency, quote_currency):
    assert FRED_FX_SERIES[series_id] == (base_currency, quote_currency)

    observation = parse_fred_observations(series_id, fred_payload())[0]

    assert (observation.base_currency, observation.quote_currency) == (
        base_currency,
        quote_currency,
    )
    assert observation.quote_per_base == 1.08
    assert observation.source == f"{FRED_SOURCE}:{series_id}"
    assert observation.price_type is PriceType.REFERENCE


def test_missing_dot_or_blank_values_are_skipped_without_fabricating_observations():
    payload = {
        "observations": [
            {"date": "2025-01-02", "value": "."},
            {"date": "2025-01-03", "value": ""},
            {"date": "2025-01-06", "value": "1.09"},
        ]
    }

    observations = parse_fred_observations("DEXUSEU", payload)

    assert len(observations) == 1
    assert observations[0].market_date == date(2025, 1, 6)


def test_duplicate_fred_observations_are_rejected():
    payload = {
        "observations": [
            {"date": "2025-01-02", "value": "1.08"},
            {"date": "2025-01-02", "value": "1.09"},
        ]
    }

    with pytest.raises(FredResponseError, match="Duplicate observation"):
        parse_fred_observations("DEXUSEU", payload)


@pytest.mark.parametrize("value", ["0", "-1", "NaN", "inf", "-inf", "not-a-number"])
def test_invalid_rates_are_rejected(value):
    with pytest.raises(FredResponseError):
        parse_fred_observations("DEXUSEU", fred_payload(value=value))


def test_unknown_series_is_rejected_without_a_request():
    session = FakeSession([])
    client = FredClient(session=session)

    with pytest.raises(ValueError, match="Unsupported FRED FX series"):
        client.fetch_series("UNKNOWN")

    assert session.calls == []


@pytest.mark.parametrize("payload", [None, [], {}, {"observations": "invalid"}])
def test_invalid_response_shapes_are_rejected(payload):
    with pytest.raises(FredResponseError):
        parse_fred_observations("DEXUSEU", payload)


def test_invalid_json_is_reported(monkeypatch):
    monkeypatch.setenv("FRED_API_KEY", "test-key")
    client = FredClient(
        session=FakeSession([FakeResponse(json_error=ValueError("invalid json"))]),
        retry_backoff_seconds=0,
    )
    with pytest.raises(FredResponseError, match="invalid JSON"):
        client.fetch_series("DEXUSEU")


def test_missing_api_key_fails_before_http_request(monkeypatch):
    monkeypatch.delenv("FRED_API_KEY", raising=False)
    session = FakeSession([])
    client = FredClient(session=session)

    with pytest.raises(FredConfigurationError, match="FRED_API_KEY"):
        client.fetch_series("DEXUSEU")

    assert session.calls == []


def test_network_error_retries_only_up_to_configured_attempts(monkeypatch):
    monkeypatch.setenv("FRED_API_KEY", "test-key")
    session = FakeSession([requests.Timeout(), requests.ConnectionError()])
    client = FredClient(session=session, max_attempts=2, retry_backoff_seconds=0)

    with pytest.raises(FredNetworkError, match="DEXUSEU"):
        client.fetch_series("DEXUSEU")

    assert len(session.calls) == 2


def test_temporary_http_error_is_retried(monkeypatch):
    monkeypatch.setenv("FRED_API_KEY", "test-key")
    session = FakeSession([FakeResponse(status_code=503), FakeResponse(fred_payload())])
    client = FredClient(session=session, max_attempts=2, retry_backoff_seconds=0)

    observations = client.fetch_series("DEXUSEU")

    assert len(session.calls) == 2
    assert observations[0].quote_per_base == 1.08


def test_fetch_raw_series_paginates_and_preserves_logical_request_params(monkeypatch):
    monkeypatch.setenv("FRED_API_KEY", "test-key")
    first_page = FakeResponse(
        {
            "count": "3",
            "offset": 0,
            "limit": 2,
            "observations": [
                {"date": "2025-01-02", "value": "1.08"},
                {"date": "2025-01-03", "value": "1.09"},
            ],
        }
    )
    second_page = FakeResponse(
        {
            "count": "3",
            "offset": 2,
            "limit": 2,
            "observations": [{"date": "2025-01-06", "value": "1.10"}],
        }
    )
    session = FakeSession([first_page, second_page])
    client = FredClient(session=session, page_size=2, retry_backoff_seconds=0)

    response = client.fetch_raw_series("DEXUSEU")

    assert len(response.payload["observations"]) == 3
    assert response.payload["count"] == "3"
    assert response.payload["limit"] == 2
    assert response.request_params["offset"] == 0
    assert [call["params"]["offset"] for call in session.calls] == [0, 2]
    assert all(call["params"]["api_key"] == "test-key" for call in session.calls)


def test_fetch_raw_series_rejects_missing_pagination_page(monkeypatch):
    monkeypatch.setenv("FRED_API_KEY", "test-key")
    session = FakeSession(
        [
            FakeResponse(
                {
                    "count": "3",
                    "observations": [
                        {"date": "2025-01-02", "value": "1.08"},
                        {"date": "2025-01-03", "value": "1.09"},
                    ],
                }
            ),
            FakeResponse({"count": "3", "observations": []}),
        ]
    )
    client = FredClient(session=session, page_size=2, retry_backoff_seconds=0)

    with pytest.raises(FredResponseError, match="incomplete page"):
        client.fetch_raw_series("DEXUSEU")


def test_fetch_raw_series_rejects_count_changes_between_pages(monkeypatch):
    monkeypatch.setenv("FRED_API_KEY", "test-key")
    session = FakeSession(
        [
            FakeResponse(
                {
                    "count": "3",
                    "observations": [
                        {"date": "2025-01-02", "value": "1.08"},
                        {"date": "2025-01-03", "value": "1.09"},
                    ],
                }
            ),
            FakeResponse(
                {
                    "count": "4",
                    "offset": 2,
                    "observations": [
                        {"date": "2025-01-06", "value": "1.10"},
                        {"date": "2025-01-07", "value": "1.11"},
                    ],
                }
            ),
        ]
    )
    client = FredClient(session=session, page_size=2, retry_backoff_seconds=0)

    with pytest.raises(FredResponseError, match="count changed"):
        client.fetch_raw_series("DEXUSEU")


def test_non_temporary_http_error_is_not_retried(monkeypatch):
    monkeypatch.setenv("FRED_API_KEY", "test-key")
    session = FakeSession([FakeResponse(status_code=401)])
    client = FredClient(session=session, max_attempts=3, retry_backoff_seconds=0)

    with pytest.raises(FredHTTPError, match="HTTP 401"):
        client.fetch_series("DEXUSEU")

    assert len(session.calls) == 1


def test_fetch_passes_requested_dates_timeout_and_api_key(monkeypatch):
    monkeypatch.setenv("FRED_API_KEY", "test-key")
    session = FakeSession([FakeResponse(fred_payload())])
    client = FredClient(session=session, timeout_seconds=7, retry_backoff_seconds=0)

    client.fetch_series(
        "DEXUSEU",
        start_date=date(2025, 1, 1),
        end_date=date(2025, 1, 31),
    )

    request = session.calls[0]
    assert request["url"] == FRED_API_URL
    assert request["timeout"] == 7
    assert request["params"]["observation_start"] == "2025-01-01"
    assert request["params"]["observation_end"] == "2025-01-31"
    assert request["params"]["series_id"] == "DEXUSEU"
    assert request["params"]["api_key"] == "test-key"
    assert request["params"]["file_type"] == "json"


def test_raw_fetch_exposes_complete_payload_without_api_key(monkeypatch):
    monkeypatch.setenv("FRED_API_KEY", "test-key")
    payload = {
        "count": "2",
        "offset": 0,
        "limit": 100000,
        "realtime_start": "2025-01-03",
        "realtime_end": "2025-01-03",
        "observations": [
            {"date": "2025-01-02", "value": "1.08"},
            {"date": "2025-01-03", "value": "."},
        ],
    }
    session = FakeSession([FakeResponse(payload)])
    client = FredClient(session=session, retry_backoff_seconds=0)

    response = client.fetch_raw_series("DEXUSEU")

    assert response.payload == payload
    assert response.request_params["series_id"] == "DEXUSEU"
    assert "api_key" not in response.request_params
    assert response.retrieved_at.utcoffset().total_seconds() == 0


@pytest.mark.parametrize(
    ("start_date", "end_date"),
    [
        (date(2025, 1, 3), date(2025, 1, 2)),
    ],
)
def test_requested_date_range_is_validated(start_date, end_date):
    with pytest.raises(ValueError, match="start_date must not be after end_date"):
        parse_fred_observations(
            "DEXUSEU",
            fred_payload(),
            start_date=start_date,
            end_date=end_date,
        )


def test_timestamp_is_not_accepted_as_date_range():
    with pytest.raises(TypeError, match="start_date"):
        parse_fred_observations(
            "DEXUSEU",
            fred_payload(),
            start_date=datetime(2025, 1, 1),
        )


def test_observations_outside_requested_range_are_rejected():
    with pytest.raises(FredResponseError, match="before the requested start date"):
        parse_fred_observations(
            "DEXUSEU",
            fred_payload(date_string="2024-12-31"),
            start_date=date(2025, 1, 1),
        )


def test_fred_availability_is_not_invented():
    observation = parse_fred_observations("DEXUSEU", fred_payload())[0]

    assert observation.available_at is None
    assert observation.availability_known is False


def test_response_with_truncated_observation_count_is_rejected(monkeypatch):
    monkeypatch.setenv("FRED_API_KEY", "test-key")
    payload = {"count": "2", "observations": [{"date": "2025-01-02", "value": "1.08"}]}
    session = FakeSession([FakeResponse(payload), FakeResponse({"count": "2", "observations": []})])
    client = FredClient(session=session, retry_backoff_seconds=0)

    with pytest.raises(FredResponseError, match="incomplete"):
        client.fetch_series("DEXUSEU")
