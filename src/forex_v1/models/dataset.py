"""Causal, development-only supervised dataset preparation (no model training)."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from bisect import bisect_right
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import date, timedelta
from itertools import pairwise
from pathlib import Path
from typing import Any

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import yaml

from forex_v1.backtest.data import MarketData, load_market_data
from forex_v1.data.layout import PROJECT_ROOT
from forex_v1.data.storage._files import write_immutable_file
from forex_v1.models.temporal import (
    FoldStandardizer,
    WalkForwardFold,
    build_walk_forward_folds,
)

DATASET_VERSION = "1"
DEFAULT_FEATURE_NAMES = (
    "log_return_1",
    "log_return_5",
    "log_return_20",
    "log_return_60",
    "rsi_14",
    "sma_relative_20",
    "sma_relative_50",
    "sma_relative_200",
    "volatility_20",
    "volatility_60",
    "bollinger_zscore_20",
    "channel_position_20",
    "drawdown_60",
)
FOREIGN_CURRENCIES = ("USD", "GBP", "JPY", "CHF", "CAD", "AUD", "NZD")
CURRENCY_ORDER = ("EUR",) + FOREIGN_CURRENCIES
DATASET_HASH_KEY = b"forex_v1.ml_dataset.content_sha256"
DATASET_VERSION_KEY = b"forex_v1.ml_dataset.version"


class MLDatasetError(ValueError):
    """Raised when a causal development dataset cannot safely be constructed."""


@dataclass(frozen=True, slots=True)
class MLDatasetConfig:
    """Validated configuration for dataset construction and walk-forward folds."""

    development_fraction: float
    availability_lag_days: int
    feature_names: tuple[str, ...]
    fold_count: int
    minimum_training_years: int
    embargo_days: int

    def __post_init__(self) -> None:
        if (
            isinstance(self.development_fraction, bool)
            or not isinstance(self.development_fraction, (int, float))
            or not math.isfinite(float(self.development_fraction))
            or not 0 < self.development_fraction < 1
        ):
            raise MLDatasetError("development_fraction must be finite and strictly between 0 and 1")
        if (
            isinstance(self.availability_lag_days, bool)
            or not isinstance(self.availability_lag_days, int)
            or self.availability_lag_days < 7
        ):
            raise MLDatasetError("availability_lag_days must be an integer of at least 7 days")
        if not self.feature_names or len(set(self.feature_names)) != len(self.feature_names):
            raise MLDatasetError("feature_names must be a non-empty ordered list without duplicates")
        if any(name.startswith("momentum_") for name in self.feature_names):
            raise MLDatasetError("momentum_* features duplicate the corresponding log_return_* inputs")
        for field_name, value, minimum in (
            ("fold_count", self.fold_count, 2),
            ("minimum_training_years", self.minimum_training_years, 1),
            ("embargo_days", self.embargo_days, 0),
        ):
            if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
                raise MLDatasetError(f"{field_name} must be an integer greater than or equal to {minimum}")


@dataclass(frozen=True, slots=True)
class DatasetBuildResult:
    """Development examples and auditable counts produced from one validated history."""

    rows: tuple[dict[str, Any], ...]
    feature_columns: tuple[str, ...]
    target_columns: tuple[str, ...]
    split_index: int
    development_end: date
    test_start: date
    market_observation_count: int
    candidate_signal_count: int
    exclusion_counts: Mapping[str, int]
    first_candidate_signal: date
    last_candidate_signal: date
    configuration: MLDatasetConfig
    eur_source_file: str
    eur_source_sha256: str
    feature_source_file: str
    feature_source_sha256: str

    @property
    def examples(self) -> pd.DataFrame:
        """Return examples as a frame while keeping the stored records immutable."""
        return pd.DataFrame.from_records(self.rows)


def _feature_columns(feature_names: Sequence[str]) -> tuple[str, ...]:
    return tuple(
        f"{currency}__{feature_name}"
        for currency in FOREIGN_CURRENCIES
        for feature_name in feature_names
    )


def _target_columns() -> tuple[str, ...]:
    return tuple(f"y_simple_return_{currency}" for currency in CURRENCY_ORDER)


def _signal_dates(start: date, end: date) -> tuple[date, ...]:
    first = start + timedelta(days=(4 - start.weekday()) % 7)
    signals: list[date] = []
    while first <= end:
        signals.append(first)
        first += timedelta(days=7)
    return tuple(signals)


def _eligible_feature_values(
    market_data: MarketData,
    cutoff: date,
    feature_names: Sequence[str],
    feature_dates: Sequence[date],
) -> tuple[date | None, dict[str, float] | None]:
    """Read the latest feature row dated no later than the signal's conservative cutoff."""
    position = bisect_right(feature_dates, cutoff) - 1
    if position < 0:
        return None, None
    market_date = feature_dates[position]
    source = market_data.features_by_date.get(market_date, {})
    values: dict[str, float] = {}
    for column in _feature_columns(feature_names):
        value = source.get(column)
        if value is None:
            return market_date, None
        numeric = float(value)
        if not math.isfinite(numeric):
            return market_date, None
        values[column] = numeric
    return market_date, values


def build_ml_dataset(
    market_data: MarketData,
    configuration: MLDatasetConfig,
) -> DatasetBuildResult:
    """Build X and execution-to-execution Y using development data only.

    Development membership is determined from the common price-calendar observation
    count before any feature warm-up or label filtering. X is read as of signal minus
    the configured calendar-day lag. Labels use prices only at the current and next
    scheduled, non-colliding weekly executions inside development.
    """
    if not market_data.feature_source_file or not market_data.feature_source_sha256:
        raise MLDatasetError("A validated feature Parquet is required")
    missing_features = set(configuration.feature_names) - set(market_data.feature_names)
    if missing_features:
        raise MLDatasetError(f"Feature history is missing columns: {sorted(missing_features)}")
    if len(market_data.dates) < 2:
        raise MLDatasetError("At least two common market observations are required for a split")

    split_index = int(len(market_data.dates) * configuration.development_fraction)
    if split_index < 2 or split_index >= len(market_data.dates):
        raise MLDatasetError("The configured chronological split leaves an empty partition")
    development_dates = market_data.dates[:split_index]
    development_end = development_dates[-1]
    test_start = market_data.dates[split_index]
    signals = _signal_dates(development_dates[0], development_end)

    # A later signal that maps to an already scheduled observation is not a second
    # execution, matching the backtest's execution-collision rule.
    execution_by_signal: dict[date, date | None] = {}
    last_execution: date | None = None
    for signal in signals:
        index = bisect_right(development_dates, signal)
        execution = development_dates[index] if index < len(development_dates) else None
        if execution is not None and execution == last_execution:
            execution = None
        execution_by_signal[signal] = execution
        if execution is not None:
            last_execution = execution
    executable_signals = [signal for signal in signals if execution_by_signal[signal] is not None]
    next_execution_by_signal = {
        signal: execution_by_signal[next_signal]
        for signal, next_signal in pairwise(executable_signals)
    }

    feature_dates = tuple(market_date for market_date in market_data.dates if market_date <= development_end)
    columns = _feature_columns(configuration.feature_names)
    targets = _target_columns()
    exclusions: Counter[str] = Counter()
    rows: list[dict[str, Any]] = []

    for signal in signals:
        cutoff = signal - timedelta(days=configuration.availability_lag_days)
        information_date, feature_values = _eligible_feature_values(
            market_data,
            cutoff,
            configuration.feature_names,
            feature_dates,
        )
        if information_date is None:
            exclusions["no_information_observation"] += 1
            continue
        if feature_values is None:
            exclusions["missing_or_nonfinite_features"] += 1
            continue
        execution = execution_by_signal[signal]
        if execution is None:
            exclusions["no_execution_or_execution_collision"] += 1
            continue
        next_execution = next_execution_by_signal.get(signal)
        if next_execution is None:
            exclusions["no_next_execution_within_development"] += 1
            continue
        if next_execution > development_end or next_execution >= test_start:
            exclusions["label_outside_development"] += 1
            continue

        start_prices = market_data.prices_by_date[execution]
        end_prices = market_data.prices_by_date[next_execution]
        row: dict[str, Any] = {
            "signal_date": signal,
            "information_cutoff_date": cutoff,
            "last_feature_market_date": information_date,
            "execution_date": execution,
            "next_execution_date": next_execution,
        }
        row.update(feature_values)
        for currency in CURRENCY_ORDER:
            start_price = float(start_prices[currency])
            end_price = float(end_prices[currency])
            row[f"price_start_eur_per_unit_{currency}"] = start_price
            row[f"price_end_eur_per_unit_{currency}"] = end_price
            row[f"y_simple_return_{currency}"] = end_price / start_price - 1.0
        rows.append(row)

    if not rows:
        raise MLDatasetError("No complete, execution-aligned development examples were produced")
    if any(row["next_execution_date"] >= test_start for row in rows):
        raise MLDatasetError("A target label crosses the protected final-test boundary")
    if any(
        row["last_feature_market_date"] > row["information_cutoff_date"]
        for row in rows
    ):
        raise MLDatasetError("An input feature was dated after its information cutoff")
    if any(row["y_simple_return_EUR"] != 0.0 for row in rows):
        raise MLDatasetError("The EUR reference-currency target must be exactly zero")
    if sum(exclusions.values()) + len(rows) != len(signals):
        raise MLDatasetError("Candidate-signal counts do not reconcile")

    return DatasetBuildResult(
        rows=tuple(rows),
        feature_columns=columns,
        target_columns=targets,
        split_index=split_index,
        development_end=development_end,
        test_start=test_start,
        market_observation_count=len(market_data.dates),
        candidate_signal_count=len(signals),
        exclusion_counts=dict(sorted(exclusions.items())),
        first_candidate_signal=signals[0],
        last_candidate_signal=signals[-1],
        configuration=configuration,
        eur_source_file=market_data.eur_source_file,
        eur_source_sha256=market_data.eur_source_sha256,
        feature_source_file=market_data.feature_source_file,
        feature_source_sha256=market_data.feature_source_sha256,
    )


def _json_date(value: Any) -> str:
    if isinstance(value, (date, pd.Timestamp)):
        return value.isoformat()
    raise TypeError(f"Unsupported date value {value!r}")


def dataset_manifest(result: DatasetBuildResult) -> dict[str, Any]:
    """Return deterministic provenance, target, feature and split metadata."""
    return {
        "dataset_version": DATASET_VERSION,
        "development_fraction": result.configuration.development_fraction,
        "development_observation_count": result.split_index,
        "market_observation_count_used_for_split": result.market_observation_count,
        "development_end": result.development_end.isoformat(),
        "test_start_excluded": result.test_start.isoformat(),
        "candidate_signal_count": result.candidate_signal_count,
        "example_count": len(result.rows),
        "first_signal_date": min(row["signal_date"] for row in result.rows).isoformat(),
        "last_signal_date": max(row["signal_date"] for row in result.rows).isoformat(),
        "availability_lag_days": result.configuration.availability_lag_days,
        "walk_forward": {
            "method": "expanding window with outcome-end purge",
            "fold_count": result.configuration.fold_count,
            "minimum_training_years": result.configuration.minimum_training_years,
            "embargo_days": result.configuration.embargo_days,
            "training_label_condition": (
                "next_execution_date + embargo_days < first_validation_signal"
            ),
        },
        "normalization": {
            "method": "per-column standard score, fit on each fold's training X only",
            "standard_deviation": "population (ddof=0)",
            "constant_column_scale": 1.0,
            "targets_scaled": False,
        },
        "feature_names": list(result.configuration.feature_names),
        "feature_columns": list(result.feature_columns),
        "target_columns": list(result.target_columns),
        "target_definition": (
            "Simple EUR-per-unit price return from execution_date to next_execution_date; "
            "EUR target is exactly zero."
        ),
        "execution_rule": (
            "First common market observation strictly after a calendar Friday; next target "
            "endpoint is the next non-colliding scheduled weekly execution."
        ),
        "fred_availability": (
            "Actual publication availability is unknown. A seven-calendar-day lag is a "
            "conservative research assumption, not a certified point-in-time guarantee."
        ),
        "excluded_test_period_used": False,
        "exclusion_counts": dict(result.exclusion_counts),
        "input_sources": {
            "eur_file": result.eur_source_file,
            "eur_sha256": result.eur_source_sha256,
            "feature_file": result.feature_source_file,
            "feature_sha256": result.feature_source_sha256,
        },
    }


def _canonical_dataset_bytes(
    rows: Sequence[Mapping[str, Any]],
    manifest_without_hash: Mapping[str, Any],
) -> bytes:
    payload = {
        "manifest": manifest_without_hash,
        "rows": [
            {
                key: (_json_date(value) if isinstance(value, (date, pd.Timestamp)) else value)
                for key, value in row.items()
            }
            for row in rows
        ],
    }
    return json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")


def _write_parquet_immutable(
    path: Path,
    rows: Sequence[Mapping[str, Any]],
    *,
    content_hash: str,
    manifest: Mapping[str, Any],
) -> None:
    if path.exists():
        try:
            existing = pq.read_table(path)
            metadata = existing.schema.metadata or {}
            if metadata.get(DATASET_HASH_KEY, b"").decode("ascii") != content_hash:
                raise MLDatasetError(f"Existing dataset artifact has conflicting content: {path}")
            existing_hash = hashlib.sha256(
                _canonical_dataset_bytes(existing.to_pylist(), manifest)
            ).hexdigest()
            if existing_hash != content_hash:
                raise MLDatasetError(f"Existing dataset artifact failed its content check: {path}")
            return
        except (OSError, pa.ArrowException, UnicodeDecodeError) as error:
            raise MLDatasetError(f"Cannot verify existing dataset artifact {path}: {error}") from error
    table = pa.Table.from_pylist([dict(row) for row in rows])
    metadata = dict(table.schema.metadata or {})
    metadata[DATASET_HASH_KEY] = content_hash.encode("ascii")
    metadata[DATASET_VERSION_KEY] = DATASET_VERSION.encode("ascii")
    sink = pa.BufferOutputStream()
    pq.write_table(table.replace_schema_metadata(metadata), sink, compression="zstd", version="2.6")
    write_immutable_file(path, sink.getvalue().to_pybytes())
    try:
        written = pq.read_table(path)
        written_metadata = written.schema.metadata or {}
        written_hash = hashlib.sha256(
            _canonical_dataset_bytes(written.to_pylist(), manifest)
        ).hexdigest()
    except (OSError, pa.ArrowException, UnicodeDecodeError) as error:
        raise MLDatasetError(f"Cannot verify written dataset artifact {path}: {error}") from error
    if (
        written_metadata.get(DATASET_HASH_KEY, b"").decode("ascii") != content_hash
        or written_hash != content_hash
    ):
        raise MLDatasetError(f"Written dataset artifact failed its content check: {path}")


def export_ml_dataset(
    result: DatasetBuildResult,
    output_dir: str | Path,
    *,
    folds: Sequence[WalkForwardFold],
) -> tuple[Path, Path, tuple[Path, ...]]:
    """Write immutable dataset, manifest, and fold-specific train-only transforms."""
    root = Path(output_dir)
    root.mkdir(parents=True, exist_ok=True)
    manifest = dataset_manifest(result)
    digest = hashlib.sha256(_canonical_dataset_bytes(result.rows, manifest)).hexdigest()
    manifest["content_sha256"] = digest
    stem = f"ml_dataset_v{DATASET_VERSION}_{digest}"
    dataset_path = root / f"{stem}.parquet"
    manifest_path = root / f"{stem}.json"
    _write_parquet_immutable(
        dataset_path,
        result.rows,
        content_hash=digest,
        manifest={key: value for key, value in manifest.items() if key != "content_sha256"},
    )
    write_immutable_file(
        manifest_path,
        (json.dumps(manifest, indent=2, sort_keys=True, allow_nan=False) + "\n").encode("utf-8"),
    )

    examples = result.examples
    exported: list[Path] = []
    fold_root = root / "folds" / stem
    for fold in folds:
        train = examples.iloc[list(fold.train_indices)]
        validation = examples.iloc[list(fold.validation_indices)]
        scaler = FoldStandardizer.fit(train.loc[:, result.feature_columns])
        train_x = scaler.transform(train.loc[:, result.feature_columns])
        validation_x = scaler.transform(validation.loc[:, result.feature_columns])
        for label, source, normalized in (
            ("train", train, train_x),
            ("validation", validation, validation_x),
        ):
            frame = pd.concat(
                [
                    source.loc[:, ["signal_date", "execution_date", "next_execution_date"]]
                    .reset_index(drop=True),
                    normalized.reset_index(drop=True),
                    source.loc[:, result.target_columns].reset_index(drop=True),
                ],
                axis=1,
            )
            csv_path = fold_root / f"fold_{fold.fold_number:02d}_{label}.csv"
            write_immutable_file(
                csv_path,
                frame.to_csv(
                    index=False,
                    float_format="%.17g",
                    lineterminator="\n",
                ).encode("utf-8"),
            )
        scaler_path = fold_root / f"fold_{fold.fold_number:02d}_scaler.json"
        scaler_payload = {
            **scaler.to_dict(),
            "fold_number": fold.fold_number,
            "training_start": fold.training_start.isoformat(),
            "training_end_signal": fold.training_end_signal.isoformat(),
            "validation_start": fold.validation_start.isoformat(),
            "validation_end_signal": fold.validation_end_signal.isoformat(),
            "purged_training_examples": fold.purged_training_examples,
            "embargo_days": fold.embargo_days,
            "fit_scope": "training X rows only",
        }
        write_immutable_file(
            scaler_path,
            (json.dumps(scaler_payload, indent=2, sort_keys=True, allow_nan=False) + "\n").encode(
                "utf-8"
            ),
        )
        exported.extend(
            [
                fold_root / f"fold_{fold.fold_number:02d}_train.csv",
                fold_root / f"fold_{fold.fold_number:02d}_validation.csv",
                scaler_path,
            ]
        )
    return dataset_path, manifest_path, tuple(exported)


def load_ml_dataset_config(path: str | Path) -> MLDatasetConfig:
    """Load and validate the dedicated dataset YAML."""
    try:
        document = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError) as error:
        raise MLDatasetError(f"Cannot read ML dataset configuration {path}: {error}") from error
    if not isinstance(document, Mapping):
        raise MLDatasetError("ML dataset config root must be a mapping")
    features = document.get("feature_names")
    if not isinstance(features, list) or any(not isinstance(item, str) for item in features):
        raise MLDatasetError("feature_names must be a list of strings")
    return MLDatasetConfig(
        development_fraction=document.get("development_fraction"),
        availability_lag_days=document.get("availability_lag_days"),
        feature_names=tuple(features),
        fold_count=document.get("fold_count"),
        minimum_training_years=document.get("minimum_training_years"),
        embargo_days=document.get("embargo_days"),
    )


def run_ml_dataset_pipeline(
    *,
    config_path: str | Path,
    eur_data_path: str | Path,
    feature_data_path: str | Path,
    output_dir: str | Path,
    report_path: str | Path,
) -> dict[str, Any]:
    """Build and export the development dataset and deterministic validation report."""
    configuration = load_ml_dataset_config(config_path)
    market_data = load_market_data(eur_data_path, feature_data_path)
    result = build_ml_dataset(market_data, configuration)
    folds = build_walk_forward_folds(
        result.rows,
        fold_count=configuration.fold_count,
        minimum_training_years=configuration.minimum_training_years,
        embargo_days=configuration.embargo_days,
    )
    dataset_path, manifest_path, fold_paths = export_ml_dataset(
        result,
        output_dir,
        folds=folds,
    )
    report = _render_validation_report(result, folds, dataset_path, manifest_path, fold_paths)
    report_file = Path(report_path)
    report_file.parent.mkdir(parents=True, exist_ok=True)
    report_file.write_text(report, encoding="utf-8")
    return {
        "result": result,
        "folds": folds,
        "dataset_path": dataset_path,
        "manifest_path": manifest_path,
        "fold_paths": fold_paths,
        "report_path": report_file,
    }


def _render_validation_report(
    result: DatasetBuildResult,
    folds: Sequence[WalkForwardFold],
    dataset_path: Path,
    manifest_path: Path,
    fold_paths: Sequence[Path],
) -> str:
    first = result.rows[0]
    last = result.rows[-1]
    lines = [
        "# ML dataset validation (step 8A)",
        "",
        "## Scope and protected boundary",
        "",
        (
            "This is a supervised dataset preparation and temporal-validation utility only. "
            "No model is defined, fitted, selected, or evaluated."
        ),
        "",
        f"- Common market observations: {result.market_observation_count:,}.",
        (
            f"- Development observations: {result.split_index:,} "
            f"({result.configuration.development_fraction:.0%})."
        ),
        f"- Development boundary: **{result.development_end.isoformat()}** inclusive.",
        (
            f"- Final-test period begins **{result.test_start.isoformat()}** and was excluded "
            "from inputs, labels, folds, and normalization."
        ),
        (
            f"- Availability assumption: {result.configuration.availability_lag_days} calendar "
            "days; FRED publication dates remain unknown."
        ),
        "",
        "## Dataset",
        "",
        f"- Candidate Friday signals: {result.candidate_signal_count:,}.",
        f"- Retained examples: {len(result.rows):,}.",
        (
            f"- First retained signal: {first['signal_date']} (cutoff "
            f"{first['information_cutoff_date']}, last feature market date "
            f"{first['last_feature_market_date']}, execution {first['execution_date']})."
        ),
        (
            f"- Last retained signal: {last['signal_date']} (cutoff "
            f"{last['information_cutoff_date']}, last feature market date "
            f"{last['last_feature_market_date']}, execution {last['execution_date']}, "
            f"next execution {last['next_execution_date']})."
        ),
        (
            f"- X: {len(result.feature_columns)} ordered currency/feature columns across "
            f"{len(result.configuration.feature_names)} indicators."
        ),
        (
            "- Y: simple EUR-per-unit returns between the actual execution date and the next "
            "scheduled, non-colliding weekly execution; EUR return is exactly zero."
        ),
        (
            "- Redundant `momentum_*` features and raw prices are not part of X. Prices are "
            "retained only as separate label-audit fields."
        ),
        "",
        "### Candidate exclusions",
        "",
        "| Reason | Count |",
        "|---|---:|",
    ]
    for reason, count in result.exclusion_counts.items():
        lines.append(f"| {reason} | {count} |")
    lines.extend(
        [
            "",
            "## Purged expanding-window folds",
            "",
            (
                f"- Folds: {len(folds)}; minimum initial training span: "
                f"{result.configuration.minimum_training_years} calendar years."
            ),
            f"- Embargo after training labels: {result.configuration.embargo_days} days.",
            (
                "- Any training sample whose `next_execution_date + embargo` is not strictly "
                "before the validation start is purged."
            ),
            (
                "- Each fold's scaler is fitted only on that fold's training X. Validation X "
                "is transformed with those stored training statistics; Y is not scaled."
            ),
            "",
            "| Fold | Train examples | Validation examples | Train signals | Validation signals | Purged |",
            "|---:|---:|---:|---|---|---:|",
        ]
    )
    for fold in folds:
        lines.append(
            f"| {fold.fold_number} | {len(fold.train_indices)} | "
            f"{len(fold.validation_indices)} | {fold.training_start}..{fold.training_end_signal} | "
            f"{fold.validation_start}..{fold.validation_end_signal} | "
            f"{fold.purged_training_examples} |"
        )
    lines.extend(
        [
            "",
            "## Reproducibility and limitations",
            "",
            f"- Dataset: `{dataset_path}`.",
            f"- Manifest (includes input SHA-256 values): `{manifest_path}`.",
            f"- Fold artifacts: {len(fold_paths)} immutable files.",
            (
                "- Market dates do not establish real-time publication availability. The seven-day "
                "lag is a conservative research convention, not proof of point-in-time availability."
            ),
            (
                "- EUR-converted FRED observations are indicative reference rates, not certified "
                "executable prices."
            ),
            (
                "- Targets are observed price returns only; there is no interest, carry, spread, "
                "slippage, or trading simulation in this dataset builder."
            ),
            (
                "- No synthetic observations, forward-fill, imputation, or use of the final-test "
                "period is performed."
            ),
            "",
        ]
    )
    return "\n".join(lines)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=PROJECT_ROOT / "configs" / "ml_dataset.yaml")
    parser.add_argument("--eur-data", default=PROJECT_ROOT / "data" / "derived" / "eur_rates")
    parser.add_argument("--feature-data", default=PROJECT_ROOT / "data" / "derived" / "features")
    parser.add_argument("--output-dir", default=PROJECT_ROOT / "data" / "derived" / "ml")
    parser.add_argument(
        "--report",
        default=PROJECT_ROOT / "reports" / "ml_dataset_validation.md",
    )
    args = parser.parse_args(argv)
    outputs = run_ml_dataset_pipeline(
        config_path=args.config,
        eur_data_path=args.eur_data,
        feature_data_path=args.feature_data,
        output_dir=args.output_dir,
        report_path=args.report,
    )
    print(f"Dataset: {outputs['dataset_path']}")
    print(f"Manifest: {outputs['manifest_path']}")
    print(f"Validation report: {outputs['report_path']}")
    print(f"Fold artifacts: {len(outputs['fold_paths'])}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
