"""Validation helpers for feature configurations and generated records."""

import math
from collections.abc import Mapping, Sequence
from datetime import date
from typing import Any

import pandas as pd

from forex_v1.features.technical import SUPPORTED_INDICATORS

MINIMUM_WINDOWS = {
    "log_return": 1,
    "rsi": 1,
    "momentum": 1,
    "sma_relative": 1,
    "volatility": 2,
    "bollinger_zscore": 2,
    "channel_position": 1,
    "drawdown": 1,
}


def validate_feature_configuration(configuration: Any) -> dict[str, Any]:
    """Validate and normalize a YAML-decoded feature configuration."""
    if not isinstance(configuration, Mapping):
        raise TypeError("Feature configuration must be a YAML mapping")
    version = configuration.get("version")
    if isinstance(version, bool) or not isinstance(version, int) or version < 1:
        raise ValueError("Feature configuration version must be a positive integer")
    annualization = configuration.get("annualization_observations")
    if isinstance(annualization, bool) or not isinstance(annualization, int) or annualization < 1:
        raise ValueError("annualization_observations must be a positive integer")
    indicators = configuration.get("indicators")
    if not isinstance(indicators, Mapping) or not indicators:
        raise ValueError("indicators must be a non-empty mapping")
    unexpected = set(indicators) - SUPPORTED_INDICATORS
    if unexpected:
        raise ValueError(f"Unsupported indicators: {sorted(unexpected)}")

    normalized: dict[str, Any] = {
        "version": version,
        "annualization_observations": annualization,
        "indicators": {},
    }
    for indicator, settings in indicators.items():
        if not isinstance(settings, Mapping):
            raise TypeError(f"{indicator} settings must be a mapping")
        enabled = settings.get("enabled")
        windows = settings.get("windows")
        if not isinstance(enabled, bool):
            raise TypeError(f"{indicator}.enabled must be a boolean")
        if not isinstance(windows, list):
            raise TypeError(f"{indicator}.windows must be a list")
        if not windows:
            raise ValueError(f"{indicator}.windows must be a non-empty list")
        checked_windows = []
        for window in windows:
            if isinstance(window, bool) or not isinstance(window, int):
                raise TypeError(f"{indicator} windows must be integers")
            if window < MINIMUM_WINDOWS[indicator]:
                raise ValueError(
                    f"{indicator} windows must be at least {MINIMUM_WINDOWS[indicator]}"
                )
            if window in checked_windows:
                raise ValueError(f"{indicator} windows must not contain duplicates")
            checked_windows.append(window)
        normalized["indicators"][indicator] = {
            "enabled": enabled,
            "windows": checked_windows,
        }
    if not any(settings["enabled"] for settings in normalized["indicators"].values()):
        raise ValueError("At least one indicator must be enabled")
    return normalized


def validate_feature_frame(
    features: pd.DataFrame,
    *,
    expected_index: Sequence[date] | None = None,
) -> None:
    """Reject invalid feature shapes, duplicate dates/columns, and infinite values."""
    if not isinstance(features, pd.DataFrame):
        raise TypeError("features must be a pandas DataFrame")
    if features.empty:
        raise ValueError("features must not be empty")
    if features.index.has_duplicates or not features.index.is_monotonic_increasing:
        raise ValueError("feature dates must be unique and chronologically sorted")
    if features.columns.has_duplicates:
        raise ValueError("feature columns must be unique")
    if expected_index is not None and list(features.index) != list(expected_index):
        raise ValueError("feature dates do not match the input dates")
    for column in features:
        values = pd.to_numeric(features[column], errors="coerce")
        if (values.notna() & ~values.map(math.isfinite)).any():
            raise ValueError(f"{column} contains a non-finite feature value")


def validate_feature_records(records: Sequence[Mapping[str, Any]]) -> None:
    """Check required output metadata and finite-or-missing feature values."""
    if not records:
        raise ValueError("feature records must not be empty")
    seen: set[tuple[date, str]] = set()
    feature_names = set(records[0]) - {
        "market_date",
        "currency",
        "missing_feature_count",
        "all_features_available",
        "source_eur_parquet",
        "source_eur_sha256",
        "source_raw_file_ids",
        "feature_config_version",
        "feature_config_sha256",
        "available_at",
        "availability_known",
    }
    if not feature_names:
        raise ValueError("feature records have no indicator columns")
    for row in records:
        key = (row["market_date"], row["currency"])
        if key in seen:
            raise ValueError(f"duplicate feature record: {key}")
        seen.add(key)
        if row["currency"] == "EUR":
            raise ValueError("EUR cash must not have synthetic market indicators")
        if row["available_at"] is not None or row["availability_known"] is not False:
            raise ValueError("FRED feature availability must remain explicitly unknown")
        if not isinstance(row["source_eur_sha256"], str) or not row["source_eur_sha256"]:
            raise ValueError("source EUR dataset hash is required")
        if not isinstance(row["feature_config_sha256"], str) or not row["feature_config_sha256"]:
            raise ValueError("feature configuration hash is required")
        missing_count = 0
        for name in feature_names:
            value = row[name]
            if value is None:
                missing_count += 1
            elif not isinstance(value, (int, float)) or isinstance(value, bool):
                raise ValueError(f"{name} must be numeric or missing")
            elif not math.isfinite(float(value)):
                raise ValueError(f"{name} must be finite or missing")
        if missing_count != row["missing_feature_count"]:
            raise ValueError("missing_feature_count does not match indicator values")
        if row["all_features_available"] is not (missing_count == 0):
            raise ValueError("all_features_available does not match indicator values")
