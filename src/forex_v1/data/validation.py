"""Validation helpers for standardized market observations."""

from collections.abc import Iterable

from forex_v1.data.schemas import FXObservation


def validate_observations(observations: Iterable[FXObservation]) -> None:
    """Reject duplicate observations from the same source and quote definition."""
    seen: set[tuple[str, str, str, object, str]] = set()
    for observation in observations:
        if not isinstance(observation, FXObservation):
            raise TypeError("observations must contain only FXObservation instances")

        key = (
            observation.source,
            observation.base_currency,
            observation.quote_currency,
            observation.market_date,
            observation.price_type.value,
        )
        if key in seen:
            raise ValueError(f"Duplicate observation detected: {key!r}")
        seen.add(key)
