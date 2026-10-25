"""FRED series-observation client for the seven configured H.10 FX series."""

import os
import time
from collections.abc import Mapping
from datetime import date, datetime
from math import isfinite
from types import MappingProxyType
from typing import Any

import requests

from forex_v1.data.schemas import FXObservation, PriceType
from forex_v1.data.validation import validate_observations

FRED_API_URL = "https://api.stlouisfed.org/fred/series/observations"
FRED_SOURCE = "fred_h10"

# FRED series are quote currency per base currency; do not invert these pairs.
FRED_FX_SERIES: Mapping[str, tuple[str, str]] = MappingProxyType(
    {
        "DEXUSEU": ("EUR", "USD"),
        "DEXUSUK": ("GBP", "USD"),
        "DEXJPUS": ("USD", "JPY"),
        "DEXSZUS": ("USD", "CHF"),
        "DEXCAUS": ("USD", "CAD"),
        "DEXUSAL": ("AUD", "USD"),
        "DEXUSNZ": ("NZD", "USD"),
    }
)

TEMPORARY_HTTP_STATUSES = frozenset({429, 500, 502, 503, 504})
DEFAULT_TIMEOUT_SECONDS = 20.0
DEFAULT_MAX_ATTEMPTS = 3
DEFAULT_RETRY_BACKOFF_SECONDS = 0.5
FRED_MAX_OBSERVATIONS = 100_000


class FredError(RuntimeError):
    """Base error for FRED configuration, transport, and response failures."""


class FredConfigurationError(FredError):
    """Raised when required FRED client configuration is missing or invalid."""


class FredNetworkError(FredError):
    """Raised when a FRED request fails due to a network or temporary HTTP error."""


class FredHTTPError(FredError):
    """Raised when FRED returns a non-retryable HTTP error."""


class FredResponseError(FredError):
    """Raised when a FRED response is malformed or contains invalid observations."""


def _validate_date_range(start_date: date | None, end_date: date | None) -> None:
    if start_date is not None and (
        not isinstance(start_date, date) or isinstance(start_date, datetime)
    ):
        raise TypeError("start_date must be a datetime.date or None")
    if end_date is not None and (
        not isinstance(end_date, date) or isinstance(end_date, datetime)
    ):
        raise TypeError("end_date must be a datetime.date or None")
    if start_date is not None and end_date is not None and start_date > end_date:
        raise ValueError("start_date must not be after end_date")


def parse_fred_observations(
    series_id: str,
    payload: Any,
    *,
    start_date: date | None = None,
    end_date: date | None = None,
) -> tuple[FXObservation, ...]:
    """Convert one FRED JSON payload to standardized observations without inventing availability."""
    if series_id not in FRED_FX_SERIES:
        raise ValueError(f"Unsupported FRED FX series: {series_id!r}")
    _validate_date_range(start_date, end_date)

    if not isinstance(payload, Mapping):
        raise FredResponseError("FRED response must be a JSON object")
    raw_observations = payload.get("observations")
    if not isinstance(raw_observations, list):
        raise FredResponseError("FRED response is missing an observations list")

    base_currency, quote_currency = FRED_FX_SERIES[series_id]
    parsed: list[FXObservation] = []
    for index, item in enumerate(raw_observations):
        if not isinstance(item, Mapping):
            raise FredResponseError(f"FRED observation at index {index} must be an object")
        raw_date = item.get("date")
        raw_value = item.get("value")
        if not isinstance(raw_date, str) or not raw_date:
            raise FredResponseError(f"FRED observation at index {index} has an invalid date")
        try:
            market_date = date.fromisoformat(raw_date)
        except ValueError as error:
            raise FredResponseError(
                f"FRED observation at index {index} has an invalid date"
            ) from error

        if start_date is not None and market_date < start_date:
            raise FredResponseError("FRED returned an observation before the requested start date")
        if end_date is not None and market_date > end_date:
            raise FredResponseError("FRED returned an observation after the requested end date")

        if raw_value in (".", ""):
            continue
        if not isinstance(raw_value, (str, int, float)) or isinstance(raw_value, bool):
            raise FredResponseError(f"FRED observation at index {index} has an invalid value")
        try:
            rate = float(raw_value)
        except (TypeError, ValueError, OverflowError) as error:
            raise FredResponseError(
                f"FRED observation at index {index} has a non-numeric value"
            ) from error
        if not isfinite(rate) or rate <= 0:
            raise FredResponseError(
                f"FRED observation at index {index} must have a finite positive rate"
            )

        try:
            parsed.append(
                FXObservation(
                    market_date=market_date,
                    base_currency=base_currency,
                    quote_currency=quote_currency,
                    quote_per_base=rate,
                    source=f"{FRED_SOURCE}:{series_id}",
                    price_type=PriceType.REFERENCE,
                    available_at=None,
                    availability_known=False,
                )
            )
        except (TypeError, ValueError) as error:
            raise FredResponseError(
                f"FRED observation at index {index} violates the FX data contract"
            ) from error

    try:
        validate_observations(parsed)
    except (TypeError, ValueError) as error:
        raise FredResponseError(f"FRED response contains invalid observations: {error}") from error
    return tuple(parsed)


class FredClient:
    """Small, bounded-retry client for the official FRED observations endpoint."""

    def __init__(
        self,
        *,
        timeout_seconds: float = DEFAULT_TIMEOUT_SECONDS,
        max_attempts: int = DEFAULT_MAX_ATTEMPTS,
        retry_backoff_seconds: float = DEFAULT_RETRY_BACKOFF_SECONDS,
        session: Any | None = None,
    ) -> None:
        if (
            isinstance(timeout_seconds, bool)
            or not isinstance(timeout_seconds, (int, float))
            or not isfinite(timeout_seconds)
            or timeout_seconds <= 0
        ):
            raise FredConfigurationError("timeout_seconds must be greater than zero")
        if isinstance(max_attempts, bool) or not isinstance(max_attempts, int) or max_attempts < 1:
            raise FredConfigurationError("max_attempts must be a positive integer")
        if (
            isinstance(retry_backoff_seconds, bool)
            or not isinstance(retry_backoff_seconds, (int, float))
            or not isfinite(retry_backoff_seconds)
            or retry_backoff_seconds < 0
        ):
            raise FredConfigurationError("retry_backoff_seconds must not be negative")

        self._timeout_seconds = timeout_seconds
        self._max_attempts = max_attempts
        self._retry_backoff_seconds = retry_backoff_seconds
        self._session = session if session is not None else requests.Session()

    def fetch_series(
        self,
        series_id: str,
        *,
        start_date: date | None = None,
        end_date: date | None = None,
    ) -> tuple[FXObservation, ...]:
        """Fetch and convert one known FRED series within the optional observation date range."""
        if series_id not in FRED_FX_SERIES:
            raise ValueError(f"Unsupported FRED FX series: {series_id!r}")
        _validate_date_range(start_date, end_date)

        api_key = os.environ.get("FRED_API_KEY")
        if not api_key or not api_key.strip():
            raise FredConfigurationError(
                "FRED_API_KEY is not set; configure it in the process environment"
            )

        params: dict[str, str | int] = {
            "api_key": api_key,
            "file_type": "json",
            "limit": FRED_MAX_OBSERVATIONS,
            "series_id": series_id,
            "sort_order": "asc",
        }
        if start_date is not None:
            params["observation_start"] = start_date.isoformat()
        if end_date is not None:
            params["observation_end"] = end_date.isoformat()

        payload = self._request_json(series_id, params)
        self._validate_result_count(payload)
        return parse_fred_observations(
            series_id,
            payload,
            start_date=start_date,
            end_date=end_date,
        )

    def _request_json(self, series_id: str, params: Mapping[str, str | int]) -> Any:
        for attempt in range(1, self._max_attempts + 1):
            try:
                response = self._session.get(
                    FRED_API_URL,
                    params=params,
                    timeout=self._timeout_seconds,
                )
            except requests.RequestException:
                if attempt == self._max_attempts:
                    raise FredNetworkError(
                        f"Network error while requesting FRED series {series_id}"
                    ) from None
                self._sleep_before_retry(attempt)
                continue

            status_code = response.status_code
            if status_code in TEMPORARY_HTTP_STATUSES:
                if attempt == self._max_attempts:
                    raise FredNetworkError(
                        f"FRED temporarily failed for series {series_id} "
                        f"(HTTP {status_code})"
                    )
                self._sleep_before_retry(attempt)
                continue
            if status_code < 200 or status_code >= 300:
                raise FredHTTPError(
                    f"FRED rejected series {series_id} with HTTP {status_code}"
                )
            try:
                return response.json()
            except (ValueError, requests.exceptions.JSONDecodeError) as error:
                raise FredResponseError(
                    f"FRED returned invalid JSON for series {series_id}"
                ) from error

        raise FredNetworkError(f"FRED request failed for series {series_id}")

    def _sleep_before_retry(self, attempt: int) -> None:
        delay = self._retry_backoff_seconds * attempt
        if delay:
            time.sleep(delay)

    @staticmethod
    def _validate_result_count(payload: Any) -> None:
        if not isinstance(payload, Mapping):
            return
        observations = payload.get("observations")
        count = payload.get("count")
        if not isinstance(observations, list) or count is None:
            return
        try:
            result_count = int(count)
        except (TypeError, ValueError) as error:
            raise FredResponseError("FRED response has an invalid observation count") from error
        if result_count > len(observations):
            raise FredResponseError(
                "FRED response is incomplete; use a narrower date range before fetching"
            )
