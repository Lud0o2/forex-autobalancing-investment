"""Build, store, read, and summarize configured technical features."""

import hashlib
import json
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Any

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import yaml

from forex_v1.data.eur_conversion import EurConversionError, EurParquetSnapshot, read_eur_parquet
from forex_v1.data.layout import PROJECT_ROOT
from forex_v1.data.storage._files import ImmutableFileError, write_immutable_file
from forex_v1.features.technical import calculate_technical_features
from forex_v1.features.validation import (
    validate_feature_configuration,
    validate_feature_frame,
    validate_feature_records,
)

FEATURE_SCHEMA_VERSION = 1
FEATURE_TRANSFORM_VERSION = "1"
FEATURE_CURRENCIES = ("USD", "GBP", "JPY", "CHF", "CAD", "AUD", "NZD")
FEATURE_METADATA_PREFIX = "forex_v1.features."
CONTENT_SHA256_KEY = b"forex_v1.features.content_sha256"
SCHEMA_VERSION_KEY = b"forex_v1.features.schema_version"
TRANSFORM_VERSION_KEY = b"forex_v1.features.transform_version"
CONFIG_SHA256_KEY = b"forex_v1.features.config_sha256"
CONFIG_VERSION_KEY = b"forex_v1.features.config_version"
SOURCE_FILE_KEY = b"forex_v1.features.source_file"
FEATURE_COLUMNS_KEY = b"forex_v1.features.columns"

RECORD_METADATA_FIELDS = (
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
)


class FeatureStoreError(RuntimeError):
    """Raised when feature data cannot be safely built, stored, or read."""


def load_feature_configuration(path: str | Path) -> tuple[dict[str, Any], str]:
    """Load and canonicalize YAML settings, returning normalized config and SHA-256."""
    try:
        decoded = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    except OSError as error:
        raise FeatureStoreError(f"Cannot read feature configuration {path}: {error}") from error
    except yaml.YAMLError as error:
        raise FeatureStoreError(f"Feature configuration YAML is invalid: {error}") from error
    configuration = validate_feature_configuration(decoded)
    body = json.dumps(
        configuration,
        ensure_ascii=True,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return configuration, hashlib.sha256(body).hexdigest()


def _source_rows(snapshot: EurParquetSnapshot) -> dict[str, list[dict[str, Any]]]:
    grouped = {currency: [] for currency in FEATURE_CURRENCIES}
    seen: set[tuple[date, str]] = set()
    for row in snapshot.rows:
        currency = row["currency"]
        if currency not in grouped:
            continue
        key = (row["market_date"], currency)
        if key in seen:
            raise FeatureStoreError(f"Duplicate EUR input row: {key}")
        seen.add(key)
        if not math.isfinite(row["eur_per_unit"]) or row["eur_per_unit"] <= 0:
            raise FeatureStoreError(f"Invalid EUR rate in source row: {key}")
        if row["availability_known"] or row["available_at"] is not None:
            raise FeatureStoreError("FRED-derived availability must be unknown")
        grouped[currency].append(row)

    expected_dates = sorted({row["market_date"] for row in grouped["USD"]})
    if not expected_dates:
        raise FeatureStoreError("EUR source dataset has no foreign-currency observations")
    for currency, rows in grouped.items():
        rows.sort(key=lambda row: row["market_date"])
        if [row["market_date"] for row in rows] != expected_dates:
            raise FeatureStoreError(
                f"EUR source dates for {currency} are not on one complete common calendar"
            )
    return grouped


def build_feature_records(
    snapshot: EurParquetSnapshot,
    configuration: Mapping[str, Any],
    *,
    configuration_sha256: str,
) -> tuple[dict[str, Any], ...]:
    """Calculate configured causal features for all seven foreign currencies."""
    config = validate_feature_configuration(configuration)
    sources = _source_rows(snapshot)
    source_file_digest = hashlib.sha256(snapshot.path.read_bytes()).hexdigest()
    feature_records: list[dict[str, Any]] = []
    common_output_dates: list[date] | None = None
    for currency in FEATURE_CURRENCIES:
        input_rows = sources[currency]
        dates = [row["market_date"] for row in input_rows]
        if common_output_dates is None:
            common_output_dates = dates
        elif dates != common_output_dates:
            raise FeatureStoreError("Foreign currency input dates do not match")

        prices = pd.Series(
            [row["eur_per_unit"] for row in input_rows],
            index=pd.Index(dates, name="market_date"),
            dtype="float64",
            name="eur_per_unit",
        )
        calculated = calculate_technical_features(prices, config)
        validate_feature_frame(calculated, expected_index=dates)
        for position, market_date in enumerate(dates):
            values = {
                str(name): (
                    float(value)
                    if not pd.isna(value)
                    else None
                )
                for name, value in calculated.iloc[position].items()
            }
            missing_count = sum(value is None for value in values.values())
            source_row = input_rows[position]
            record: dict[str, Any] = {
                "market_date": market_date,
                "currency": currency,
                **values,
                "missing_feature_count": missing_count,
                "all_features_available": missing_count == 0,
                "source_eur_parquet": snapshot.path.name,
                "source_eur_sha256": source_file_digest,
                "source_raw_file_ids": sorted(set(source_row["source_raw_file_ids"])),
                "feature_config_version": str(config["version"]),
                "feature_config_sha256": configuration_sha256,
                "available_at": None,
                "availability_known": False,
            }
            feature_records.append(record)
    feature_records.sort(key=lambda row: (row["market_date"], row["currency"]))
    validate_feature_records(feature_records)
    return tuple(feature_records)


def _feature_names(records: Sequence[Mapping[str, Any]]) -> tuple[str, ...]:
    names = set(records[0]) - set(RECORD_METADATA_FIELDS)
    return tuple(sorted(names))


def _feature_schema(feature_names: Sequence[str]) -> pa.Schema:
    fields = [
        pa.field("market_date", pa.date32(), nullable=False),
        pa.field("currency", pa.string(), nullable=False),
    ]
    fields.extend(pa.field(name, pa.float64(), nullable=True) for name in feature_names)
    fields.extend(
        [
            pa.field("missing_feature_count", pa.int16(), nullable=False),
            pa.field("all_features_available", pa.bool_(), nullable=False),
            pa.field("source_eur_parquet", pa.string(), nullable=False),
            pa.field("source_eur_sha256", pa.string(), nullable=False),
            pa.field("source_raw_file_ids", pa.list_(pa.string()), nullable=False),
            pa.field("feature_config_version", pa.string(), nullable=False),
            pa.field("feature_config_sha256", pa.string(), nullable=False),
            pa.field("available_at", pa.timestamp("us", tz="UTC"), nullable=True),
            pa.field("availability_known", pa.bool_(), nullable=False),
        ]
    )
    return pa.schema(fields)


def _canonical_records(records: Sequence[Mapping[str, Any]], feature_names: Sequence[str]) -> bytes:
    normalized = []
    for record in records:
        row = dict(record)
        row["market_date"] = row["market_date"].isoformat()
        row["available_at"] = (
            row["available_at"].isoformat() if row["available_at"] is not None else None
        )
        normalized.append(row)
    try:
        return json.dumps(
            {"features": list(feature_names), "records": normalized},
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        ).encode("utf-8")
    except (TypeError, ValueError) as error:
        raise FeatureStoreError("Feature records cannot be serialized safely") from error


def write_feature_parquet(
    output_root: str | Path,
    records: Sequence[Mapping[str, Any]],
) -> Path:
    """Write an immutable content-addressed feature Parquet file and verify its readback."""
    validate_feature_records(records)
    features = _feature_names(records)
    config_hashes = {row["feature_config_sha256"] for row in records}
    config_versions = {row["feature_config_version"] for row in records}
    source_files = {row["source_eur_parquet"] for row in records}
    source_hashes = {row["source_eur_sha256"] for row in records}
    if any(len(values) != 1 for values in (config_hashes, config_versions, source_files, source_hashes)):
        raise FeatureStoreError("Feature records mix configurations or EUR source datasets")

    body = _canonical_records(records, features)
    content_hash = hashlib.sha256(body).hexdigest()
    path = Path(output_root) / "features" / f"features_v{FEATURE_TRANSFORM_VERSION}_{content_hash}.parquet"
    metadata = {
        CONTENT_SHA256_KEY: content_hash.encode("ascii"),
        SCHEMA_VERSION_KEY: str(FEATURE_SCHEMA_VERSION).encode("ascii"),
        TRANSFORM_VERSION_KEY: FEATURE_TRANSFORM_VERSION.encode("ascii"),
        CONFIG_SHA256_KEY: next(iter(config_hashes)).encode("ascii"),
        CONFIG_VERSION_KEY: next(iter(config_versions)).encode("ascii"),
        SOURCE_FILE_KEY: next(iter(source_files)).encode("utf-8"),
        FEATURE_COLUMNS_KEY: json.dumps(features, separators=(",", ":")).encode("utf-8"),
    }
    table = pa.Table.from_pylist(
        [dict(record) for record in records],
        schema=_feature_schema(features).with_metadata(metadata),
    )
    try:
        sink = pa.BufferOutputStream()
        pq.write_table(table, sink, compression="zstd", version="2.6")
        write_immutable_file(path, sink.getvalue().to_pybytes())
    except (pa.ArrowException, ImmutableFileError, OSError, TypeError, ValueError) as error:
        raise FeatureStoreError(f"Could not safely write feature Parquet: {error}") from error
    snapshot = read_feature_parquet(path)
    if _canonical_records(snapshot.records, snapshot.feature_names) != body:
        raise FeatureStoreError("Feature Parquet readback differs from written records")
    return path


@dataclass(frozen=True, slots=True)
class FeatureParquetSnapshot:
    records: tuple[dict[str, Any], ...]
    feature_names: tuple[str, ...]
    configuration_version: str
    configuration_sha256: str
    source_eur_file: str
    schema_version: int
    transform_version: str
    path: Path


def read_feature_parquet(path: str | Path) -> FeatureParquetSnapshot:
    """Read and verify feature Parquet metadata, content hash, missing flags, and chronology."""
    parquet_path = Path(path)
    try:
        table = pq.read_table(parquet_path)
    except (OSError, pa.ArrowException) as error:
        raise FeatureStoreError(f"Could not read feature Parquet {parquet_path}: {error}") from error
    metadata = table.schema.metadata or {}
    try:
        schema_version = int(metadata[SCHEMA_VERSION_KEY].decode("ascii"))
        transform_version = metadata[TRANSFORM_VERSION_KEY].decode("ascii")
        expected_hash = metadata[CONTENT_SHA256_KEY].decode("ascii")
        configuration_hash = metadata[CONFIG_SHA256_KEY].decode("ascii")
        configuration_version = metadata[CONFIG_VERSION_KEY].decode("ascii")
        source_file = metadata[SOURCE_FILE_KEY].decode("utf-8")
        feature_names = tuple(json.loads(metadata[FEATURE_COLUMNS_KEY]))
    except (KeyError, UnicodeDecodeError, ValueError, json.JSONDecodeError) as error:
        raise FeatureStoreError("Feature Parquet metadata is invalid or incomplete") from error
    if schema_version != FEATURE_SCHEMA_VERSION or transform_version != FEATURE_TRANSFORM_VERSION:
        raise FeatureStoreError("Feature Parquet schema or transform version is unsupported")
    if not feature_names or len(set(feature_names)) != len(feature_names):
        raise FeatureStoreError("Feature Parquet has invalid indicator column metadata")
    if not table.schema.remove_metadata().equals(_feature_schema(feature_names)):
        raise FeatureStoreError("Feature Parquet columns do not match their declared schema")

    records = tuple(table.to_pylist())
    if not records or hashlib.sha256(_canonical_records(records, feature_names)).hexdigest() != expected_hash:
        raise FeatureStoreError("Feature Parquet content hash does not match its metadata")
    validate_feature_records(records)
    if any(
        row["feature_config_sha256"] != configuration_hash
        or row["feature_config_version"] != configuration_version
        or row["source_eur_parquet"] != source_file
        for row in records
    ):
        raise FeatureStoreError("Feature row provenance conflicts with Parquet metadata")
    keys = [(row["market_date"], row["currency"]) for row in records]
    if keys != sorted(keys):
        raise FeatureStoreError("Feature Parquet records are not chronologically sorted")
    return FeatureParquetSnapshot(
        records=records,
        feature_names=feature_names,
        configuration_version=configuration_version,
        configuration_sha256=configuration_hash,
        source_eur_file=source_file,
        schema_version=schema_version,
        transform_version=transform_version,
        path=parquet_path,
    )


def summarize_features(snapshot: FeatureParquetSnapshot) -> dict[str, Any]:
    """Summarize missingness, ranges, anomalies, and all-currency warm-up date."""
    summary: dict[str, Any] = {
        "row_count": len(snapshot.records),
        "currency_count": len({row["currency"] for row in snapshot.records}),
        "date_count": len({row["market_date"] for row in snapshot.records}),
        "first_date": min(row["market_date"] for row in snapshot.records),
        "last_date": max(row["market_date"] for row in snapshot.records),
        "feature_columns": {},
        "first_complete_date_all_currencies": None,
        "non_finite_anomaly_count": 0,
    }
    by_date: dict[date, dict[str, bool]] = {}
    for name in snapshot.feature_names:
        values = [row[name] for row in snapshot.records if row[name] is not None]
        summary["feature_columns"][name] = {
            "valid_count": len(values),
            "missing_count": len(snapshot.records) - len(values),
            "minimum": min(values) if values else None,
            "maximum": max(values) if values else None,
        }
    for row in snapshot.records:
        by_date.setdefault(row["market_date"], {})[row["currency"]] = row[
            "all_features_available"
        ]
        for name in snapshot.feature_names:
            value = row[name]
            if value is not None and not math.isfinite(value):
                summary["non_finite_anomaly_count"] += 1
    foreign_currencies = set(FEATURE_CURRENCIES)
    for market_date in sorted(by_date):
        available = by_date[market_date]
        if foreign_currencies.issubset(available) and all(
            available[currency] for currency in FEATURE_CURRENCIES
        ):
            summary["first_complete_date_all_currencies"] = market_date
            break
    return summary


def write_feature_report(snapshot: FeatureParquetSnapshot, path: str | Path) -> dict[str, Any]:
    """Write a Markdown summary of actual feature output and provenance."""
    summary = summarize_features(snapshot)
    lines = [
        "# Feature validation report",
        "",
        (
            "The feature rows are derived from FRED H.10 reference-rate cross conversions, "
            "not executable prices. Historical publication availability remains unknown; "
            "this dataset is not point-in-time certified."
        ),
        "",
        f"- Dates: {summary['first_date']} through {summary['last_date']}",
        f"- Rows: {summary['row_count']}",
        f"- Foreign currencies: {summary['currency_count']}",
        (
            f"- First date where all configured features are available for all currencies: "
            f"{summary['first_complete_date_all_currencies'] or 'not reached'}"
        ),
        f"- Non-finite anomalies: {summary['non_finite_anomaly_count']}",
        f"- EUR source file: `{snapshot.source_eur_file}`",
        (
            f"- Configuration version/hash: `{snapshot.configuration_version}` / "
            f"`{snapshot.configuration_sha256}`"
        ),
        (
            f"- Feature transform/schema versions: `{snapshot.transform_version}` / "
            f"`{snapshot.schema_version}`"
        ),
        "",
        "| Feature | Valid | Missing | Minimum | Maximum |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    for name, details in summary["feature_columns"].items():
        minimum = "N/A" if details["minimum"] is None else f"{details['minimum']:.10g}"
        maximum = "N/A" if details["maximum"] is None else f"{details['maximum']:.10g}"
        lines.append(
            f"| {name} | {details['valid_count']} | {details['missing_count']} "
            f"| {minimum} | {maximum} |"
        )
    lines.extend(
        [
            "",
            (
                "Initial missing values are retained for indicator warm-up. No backfill, "
                "forward-fill, interpolation, global scaler, or centered rolling window is used."
            ),
            "",
        ]
    )
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("\n".join(lines), encoding="utf-8")
    return summary


def run_feature_pipeline(
    *,
    eur_parquet_path: str | Path,
    configuration_path: str | Path,
    output_root: str | Path,
    report_path: str | Path,
) -> FeatureParquetSnapshot:
    """Read the EUR dataset, build configured indicators, persist and report them."""
    eur_parquet_path = Path(eur_parquet_path)
    if eur_parquet_path.is_dir():
        candidates = sorted(eur_parquet_path.glob("*.parquet"))
        if len(candidates) != 1:
            raise FeatureStoreError(
                "Expected exactly one EUR Parquet snapshot in "
                f"{eur_parquet_path}, found {len(candidates)}"
            )
        eur_parquet_path = candidates[0]
    try:
        eur_snapshot = read_eur_parquet(eur_parquet_path)
    except EurConversionError as error:
        raise FeatureStoreError(f"Cannot load validated EUR-rate input: {error}") from error
    configuration, configuration_sha256 = load_feature_configuration(configuration_path)
    records = build_feature_records(
        eur_snapshot,
        configuration,
        configuration_sha256=configuration_sha256,
    )
    output_path = write_feature_parquet(output_root, records)
    snapshot = read_feature_parquet(output_path)
    write_feature_report(snapshot, report_path)
    return snapshot


def main(argv: list[str] | None = None) -> int:
    import argparse

    parser = argparse.ArgumentParser(description="Build configured Forex technical features.")
    parser.add_argument(
        "--eur-parquet",
        type=Path,
        default=PROJECT_ROOT
        / "data"
        / "derived"
        / "eur_rates",
    )
    parser.add_argument("--config", type=Path, default=PROJECT_ROOT / "configs" / "features.yaml")
    parser.add_argument("--output-root", type=Path, default=PROJECT_ROOT / "data" / "derived")
    parser.add_argument(
        "--report",
        type=Path,
        default=PROJECT_ROOT / "reports" / "features_validation.md",
    )
    args = parser.parse_args(argv)
    snapshot = run_feature_pipeline(
        eur_parquet_path=args.eur_parquet,
        configuration_path=args.config,
        output_root=args.output_root,
        report_path=args.report,
    )
    summary = summarize_features(snapshot)
    print(f"Feature Parquet: {snapshot.path}")
    print(
        f"Rows: {summary['row_count']}; date range: "
        f"{summary['first_date']} through {summary['last_date']}"
    )
    print(
        "First complete feature date across all currencies: "
        f"{summary['first_complete_date_all_currencies']}"
    )
    print(f"Report: {args.report}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
