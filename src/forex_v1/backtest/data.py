"""Validated historical price and optional feature inputs for research backtests."""

import hashlib
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Any

import pandas as pd

from forex_v1.data.eur_conversion import (
    CURRENCY_ORDER,
    EurConversionError,
    read_eur_parquet,
)
from forex_v1.data.schemas import SUPPORTED_CURRENCIES
from forex_v1.features.builder import FeatureStoreError, read_feature_parquet

FOREIGN_CURRENCIES = tuple(currency for currency in CURRENCY_ORDER if currency != "EUR")


class BacktestDataError(ValueError):
    """Raised when historical inputs cannot safely support a simulation."""


@dataclass(frozen=True, slots=True)
class MarketData:
    """Common-calendar EUR prices and an optional wide feature history."""

    prices_by_date: Mapping[date, Mapping[str, float]]
    eur_source_file: str
    eur_source_sha256: str
    features_by_date: Mapping[date, Mapping[str, float | None]]
    feature_names: tuple[str, ...]
    complete_feature_dates: frozenset[date]
    feature_source_file: str | None = None
    feature_source_sha256: str | None = None

    def __post_init__(self) -> None:
        if not self.prices_by_date:
            raise BacktestDataError("Market price history must not be empty")
        dates = tuple(self.prices_by_date)
        if dates != tuple(sorted(set(dates))):
            raise BacktestDataError("Market price dates must be unique and chronological")
        for market_date, prices in self.prices_by_date.items():
            if not isinstance(market_date, date):
                raise BacktestDataError("Market price keys must be calendar dates")
            if set(prices) != SUPPORTED_CURRENCIES:
                raise BacktestDataError(
                    f"Price row for {market_date} must contain exactly the eight V1 currencies"
                )
            if prices["EUR"] != 1.0:
                raise BacktestDataError(f"EUR price must equal 1.0 on {market_date}")
            for currency, price in prices.items():
                if isinstance(price, bool) or not isinstance(price, (int, float)):
                    raise BacktestDataError(f"Invalid {currency} price on {market_date}")
                if not pd.notna(price) or not float(price) > 0 or not float(price) < float("inf"):
                    raise BacktestDataError(
                        f"Price for {currency} on {market_date} must be finite and positive"
                    )
        if not self.eur_source_file or not self.eur_source_sha256:
            raise BacktestDataError("EUR source file and SHA-256 are required")
        if bool(self.feature_source_file) != bool(self.feature_source_sha256):
            raise BacktestDataError("Feature source file and hash must be supplied together")

    @property
    def dates(self) -> tuple[date, ...]:
        return tuple(self.prices_by_date)


def _one_parquet(path: str | Path, label: str) -> Path:
    source = Path(path)
    if source.is_dir():
        candidates = sorted(source.glob("*.parquet"))
        if len(candidates) != 1:
            raise BacktestDataError(
                f"Expected exactly one {label} Parquet in {source}; found {len(candidates)}"
            )
        return candidates[0]
    return source


def _read_eur(path: str | Path) -> tuple[dict[date, dict[str, float]], str, str]:
    parquet_path = _one_parquet(path, "EUR")
    try:
        snapshot = read_eur_parquet(parquet_path)
    except (EurConversionError, OSError) as error:
        raise BacktestDataError(f"Cannot load validated EUR history: {error}") from error
    prices: dict[date, dict[str, float]] = {}
    for row in snapshot.rows:
        market_date = row["market_date"]
        currency = row["currency"]
        if currency not in SUPPORTED_CURRENCIES:
            raise BacktestDataError(f"Unsupported currency {currency!r} in EUR history")
        daily = prices.setdefault(market_date, {})
        if currency in daily:
            raise BacktestDataError(f"Duplicate EUR price for {market_date} {currency}")
        daily[currency] = float(row["eur_per_unit"])
    for market_date, daily in prices.items():
        if set(daily) != SUPPORTED_CURRENCIES:
            raise BacktestDataError(f"Incomplete eight-currency prices for {market_date}")
    return prices, parquet_path.name, hashlib.sha256(parquet_path.read_bytes()).hexdigest()


def _read_features(
    path: str | Path,
    *,
    eur_path: Path,
    eur_file: str,
    eur_sha256: str,
    market_dates: Sequence[date],
) -> tuple[
    dict[date, dict[str, float | None]],
    tuple[str, ...],
    frozenset[date],
    str,
    str,
]:
    parquet_path = _one_parquet(path, "feature")
    try:
        snapshot = read_feature_parquet(parquet_path)
    except (FeatureStoreError, OSError) as error:
        raise BacktestDataError(f"Cannot load validated features: {error}") from error
    actual_eur_hash = hashlib.sha256(eur_path.read_bytes()).hexdigest()
    if snapshot.source_eur_file != eur_file or actual_eur_hash != eur_sha256:
        raise BacktestDataError("Feature Parquet provenance does not match selected EUR input")

    expected_dates = set(market_dates)
    by_currency_date: dict[tuple[date, str], dict[str, Any]] = {}
    expected_currencies = set(FOREIGN_CURRENCIES)
    for row in snapshot.records:
        market_date = row["market_date"]
        currency = row["currency"]
        if market_date not in expected_dates or currency not in expected_currencies:
            raise BacktestDataError("Feature dates or currencies do not match EUR input")
        key = (market_date, currency)
        if key in by_currency_date:
            raise BacktestDataError(f"Duplicate feature row for {market_date} {currency}")
        if row["source_eur_sha256"] != eur_sha256:
            raise BacktestDataError("Feature row EUR-source hash does not match selected input")
        if row["availability_known"] is not False or row["available_at"] is not None:
            raise BacktestDataError("Feature availability must remain explicitly unknown")
        by_currency_date[key] = row
    features: dict[date, dict[str, float | None]] = {}
    complete_dates: set[date] = set()
    for market_date in market_dates:
        rows = {
            currency: by_currency_date.get((market_date, currency))
            for currency in FOREIGN_CURRENCIES
        }
        if any(row is None for row in rows.values()):
            raise BacktestDataError(f"Feature calendar is incomplete on {market_date}")
        if all(row["all_features_available"] for row in rows.values() if row is not None):
            complete_dates.add(market_date)
        wide: dict[str, float | None] = {}
        for currency, row in rows.items():
            assert row is not None
            for name in snapshot.feature_names:
                wide[f"{currency}__{name}"] = row[name]
        features[market_date] = wide
    return (
        features,
        snapshot.feature_names,
        frozenset(complete_dates),
        parquet_path.name,
        hashlib.sha256(parquet_path.read_bytes()).hexdigest(),
    )


def load_market_data(
    eur_parquet_path: str | Path,
    feature_parquet_path: str | Path | None = None,
) -> MarketData:
    """Read and cross-check an EUR Parquet and an optional feature Parquet."""
    eur_path = _one_parquet(eur_parquet_path, "EUR")
    prices, eur_file, eur_sha256 = _read_eur(eur_path)
    features: dict[date, dict[str, float | None]] = {}
    feature_names: tuple[str, ...] = ()
    complete_feature_dates: frozenset[date] = frozenset()
    feature_file = None
    feature_sha256 = None
    if feature_parquet_path is not None:
        (
            features,
            feature_names,
            complete_feature_dates,
            feature_file,
            feature_sha256,
        ) = _read_features(
            feature_parquet_path,
            eur_path=eur_path,
            eur_file=eur_file,
            eur_sha256=eur_sha256,
            market_dates=tuple(prices),
        )
    return MarketData(
        prices_by_date=prices,
        eur_source_file=eur_file,
        eur_source_sha256=eur_sha256,
        features_by_date=features,
        feature_names=feature_names,
        complete_feature_dates=complete_feature_dates,
        feature_source_file=feature_file,
        feature_source_sha256=feature_sha256,
    )


def market_data_from_records(
    prices_by_date: Mapping[date, Mapping[str, float]],
    *,
    features_by_date: Mapping[date, Mapping[str, float | None]] | None = None,
) -> MarketData:
    """Create validated in-memory inputs for deterministic tests."""
    return MarketData(
        prices_by_date={market_date: dict(prices) for market_date, prices in sorted(prices_by_date.items())},
        eur_source_file="synthetic-eur.parquet",
        eur_source_sha256=hashlib.sha256(b"synthetic-eur").hexdigest(),
        features_by_date={
            market_date: dict(features)
            for market_date, features in sorted((features_by_date or {}).items())
        },
        feature_names=(
            tuple(sorted(next(iter(features_by_date.values())).keys()))
            if features_by_date
            else ()
        ),
        complete_feature_dates=(
            frozenset(
                market_date
                for market_date, values in (features_by_date or {}).items()
                if all(value is not None for value in values.values())
            )
        ),
    )
