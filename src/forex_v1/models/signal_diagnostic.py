"""Development-only audit of existing Forex features and one-week return signal."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import date, timedelta
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from forex_v1.data.layout import PROJECT_ROOT
from forex_v1.models.temporal import FoldStandardizer

RIDGE_ALPHA = 10.0
INNER_VALIDATION_FRACTION = 0.20
FOREIGN_CURRENCIES = ("USD", "GBP", "JPY", "CHF", "CAD", "AUD", "NZD")
MODEL_NAMES = ("constant_mean", "ridge")


class MLSignalDiagnosticError(ValueError):
    """Raised when the immutable 8A inputs cannot support a safe signal audit."""


@dataclass(frozen=True, slots=True)
class InnerPartition:
    fold_number: int
    train: pd.DataFrame
    validation: pd.DataFrame
    train_raw_features: pd.DataFrame
    validation_raw_features: pd.DataFrame
    scaler: FoldStandardizer
    validation_start: date
    purged_training_examples: int


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def validate_finite_features(features: pd.DataFrame) -> None:
    """Reject missing and non-finite features; this diagnostic never imputes."""
    if features.empty or not features.columns.is_unique:
        raise MLSignalDiagnosticError("Feature data must be non-empty with unique columns")
    try:
        values = features.to_numpy(dtype=np.float64)
    except (TypeError, ValueError) as error:
        raise MLSignalDiagnosticError("Features must be numeric") from error
    if not np.isfinite(values).all():
        raise MLSignalDiagnosticError("Features contain missing or non-finite values")


def validate_8a_alignment(
    rows: pd.DataFrame,
    *,
    feature_columns: Sequence[str],
    target_columns: Sequence[str],
    currencies: Sequence[str] = FOREIGN_CURRENCIES,
    development_end: date,
    test_start: date,
    availability_lag_days: int,
) -> None:
    """Validate feature as-of dates and execution-to-execution labels from 8A."""
    required_dates = (
        "signal_date",
        "information_cutoff_date",
        "last_feature_market_date",
        "execution_date",
        "next_execution_date",
    )
    required = [*required_dates, *feature_columns, *target_columns]
    for currency in currencies:
        required.extend(
            (
                f"price_start_eur_per_unit_{currency}",
                f"price_end_eur_per_unit_{currency}",
            )
        )
    missing = sorted(set(required) - set(rows.columns))
    if missing:
        raise MLSignalDiagnosticError(f"8A rows are missing required fields: {missing}")
    if rows.empty:
        raise MLSignalDiagnosticError("No 8A rows were selected for the internal partitions")
    validate_finite_features(rows.loc[:, list(feature_columns)])

    for record in rows.loc[:, list(required_dates)].itertuples(index=False, name=None):
        signal_value, cutoff_value, feature_value, execution_value, endpoint_value = record
        signal = pd.Timestamp(signal_value).date()
        cutoff = pd.Timestamp(cutoff_value).date()
        feature_date = pd.Timestamp(feature_value).date()
        execution = pd.Timestamp(execution_value).date()
        endpoint = pd.Timestamp(endpoint_value).date()
        if signal > development_end or endpoint >= test_start:
            raise MLSignalDiagnosticError("An internal sample reaches the protected test period")
        if cutoff > signal - timedelta(days=availability_lag_days):
            raise MLSignalDiagnosticError("A feature information cutoff violates the configured lag")
        if feature_date > cutoff:
            raise MLSignalDiagnosticError("A feature observation occurs after its information cutoff")
        if execution <= signal or endpoint <= execution:
            raise MLSignalDiagnosticError("Future returns are not aligned to subsequent executions")

    for currency in currencies:
        start_prices = rows[f"price_start_eur_per_unit_{currency}"].to_numpy(dtype=np.float64)
        end_prices = rows[f"price_end_eur_per_unit_{currency}"].to_numpy(dtype=np.float64)
        target_name = f"y_simple_return_{currency}"
        if target_name not in target_columns:
            raise MLSignalDiagnosticError(f"8A target is absent for {currency}")
        if (start_prices <= 0).any() or (end_prices <= 0).any():
            raise MLSignalDiagnosticError(f"8A prices must be positive for {currency}")
        expected = end_prices / start_prices - 1.0
        actual = rows[target_name].to_numpy(dtype=np.float64)
        if not np.allclose(actual, expected, rtol=1e-11, atol=1e-12):
            raise MLSignalDiagnosticError(
                f"8A future-return label does not match execution prices for {currency}"
            )
    if "y_simple_return_EUR" in target_columns and not np.allclose(
        rows["y_simple_return_EUR"].to_numpy(dtype=np.float64), 0.0, atol=1e-12
    ):
        raise MLSignalDiagnosticError("The 8A EUR return target must be exactly zero")


def fit_ridge(train_features: np.ndarray, train_targets: np.ndarray, alpha: float) -> tuple[np.ndarray, np.ndarray]:
    """Fit multi-output Ridge with an intercept and a fixed L2 coefficient."""
    x = np.asarray(train_features, dtype=np.float64)
    y = np.asarray(train_targets, dtype=np.float64)
    if (
        x.ndim != 2
        or y.ndim != 2
        or len(x) != len(y)
        or len(x) == 0
        or not np.isfinite(x).all()
        or not np.isfinite(y).all()
    ):
        raise MLSignalDiagnosticError("Ridge inputs must be finite, non-empty matrices with matching rows")
    if not math.isfinite(alpha) or alpha <= 0:
        raise MLSignalDiagnosticError("Ridge alpha must be finite and positive")
    target_mean = y.mean(axis=0)
    gram = x.T @ x + alpha * np.eye(x.shape[1], dtype=np.float64)
    coefficients = np.linalg.solve(gram, x.T @ (y - target_mean))
    return coefficients, target_mean


def predict_ridge(
    features: np.ndarray,
    coefficients: np.ndarray,
    intercept: np.ndarray,
) -> np.ndarray:
    """Apply fitted multi-output Ridge coefficients."""
    x = np.asarray(features, dtype=np.float64)
    prediction = x @ coefficients + intercept
    if not np.isfinite(prediction).all():
        raise MLSignalDiagnosticError("Ridge produced non-finite predictions")
    return prediction


def _correlation(left: np.ndarray, right: np.ndarray) -> float | None:
    if len(left) < 2 or len(right) != len(left):
        return None
    if not np.isfinite(left).all() or not np.isfinite(right).all():
        raise MLSignalDiagnosticError("Correlation inputs must be finite")
    if np.ptp(left) == 0 or np.ptp(right) == 0:
        return None
    left_centered = left - left.mean()
    right_centered = right - right.mean()
    denominator = np.linalg.norm(left_centered) * np.linalg.norm(right_centered)
    if denominator == 0:
        return None
    return float(np.dot(left_centered, right_centered) / denominator)


def _spearman(left: np.ndarray, right: np.ndarray) -> float | None:
    left_ranks = pd.Series(left).rank(method="average").to_numpy(dtype=np.float64)
    right_ranks = pd.Series(right).rank(method="average").to_numpy(dtype=np.float64)
    return _correlation(left_ranks, right_ranks)


def _target_columns(target_columns: Sequence[str]) -> tuple[str, ...]:
    expected = tuple(f"y_simple_return_{currency}" for currency in FOREIGN_CURRENCIES)
    if not set(expected).issubset(target_columns):
        raise MLSignalDiagnosticError("The dataset does not contain all seven foreign-currency targets")
    return expected


def _feature_parts(feature_column: str) -> tuple[str, str]:
    currency, indicator = feature_column.split("__", maxsplit=1)
    return currency, indicator


def _load_manifest(dataset_dir: Path) -> tuple[Path, Path, dict[str, Any]]:
    manifests = sorted(dataset_dir.glob("ml_dataset_v1_*.json"))
    if len(manifests) != 1:
        raise MLSignalDiagnosticError(
            f"Expected exactly one 8A dataset manifest in {dataset_dir}; found {len(manifests)}"
        )
    manifest_path = manifests[0]
    dataset_path = manifest_path.with_suffix(".parquet")
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise MLSignalDiagnosticError(f"Cannot read the 8A manifest: {error}") from error
    if manifest.get("excluded_test_period_used") is not False:
        raise MLSignalDiagnosticError("The 8A manifest does not certify final-test exclusion")
    feature_columns = tuple(manifest.get("feature_columns", ()))
    if len(feature_columns) != 91 or len(set(feature_columns)) != 91:
        raise MLSignalDiagnosticError("The 8A manifest must contain 91 unique features")
    if len(manifest.get("feature_names", ())) != 13:
        raise MLSignalDiagnosticError("The 8A manifest must contain 13 indicator types")
    if not dataset_path.is_file():
        raise MLSignalDiagnosticError(f"The immutable 8A Parquet is missing: {dataset_path}")
    return dataset_path, manifest_path, manifest


def _fold_root(dataset_dir: Path, manifest_path: Path) -> Path:
    return dataset_dir / "folds" / manifest_path.stem


def _split_and_scale_inner_train(
    outer_train: pd.DataFrame,
    feature_columns: Sequence[str],
    outer_scaler: FoldStandardizer,
    fold_number: int,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, FoldStandardizer, date, int]:
    """Apply the exact 8C chronological tail split, endpoint purge, and inner-only scaler."""
    validation_count = max(1, math.ceil(len(outer_train) * INNER_VALIDATION_FRACTION))
    if validation_count >= len(outer_train):
        raise MLSignalDiagnosticError(f"Fold {fold_number} is too small for an inner split")
    validation_start = outer_train.iloc[-validation_count]["signal_date"].date()
    validation = outer_train.loc[
        outer_train["signal_date"].dt.date >= validation_start
    ].copy()
    eligible_train = outer_train.loc[
        outer_train["signal_date"].dt.date < validation_start
    ]
    train = eligible_train.loc[
        eligible_train["next_execution_date"].dt.date < validation_start
    ].copy()
    purged = len(eligible_train) - len(train)
    if train.empty or validation.empty:
        raise MLSignalDiagnosticError(f"Fold {fold_number} has an empty inner partition")
    if train["next_execution_date"].max().date() >= validation_start:
        raise MLSignalDiagnosticError(f"Fold {fold_number} has an unpurged inner training label")
    if validation["signal_date"].min().date() != validation_start:
        raise MLSignalDiagnosticError(f"Fold {fold_number} inner validation boundary is inconsistent")

    columns = list(feature_columns)
    raw_train_values = (
        train.loc[:, columns].to_numpy(dtype=np.float64)
        * np.asarray(outer_scaler.scales)
        + np.asarray(outer_scaler.means)
    )
    raw_validation_values = (
        validation.loc[:, columns].to_numpy(dtype=np.float64)
        * np.asarray(outer_scaler.scales)
        + np.asarray(outer_scaler.means)
    )
    train_raw = pd.DataFrame(raw_train_values, columns=columns, index=train.index)
    validation_raw = pd.DataFrame(
        raw_validation_values,
        columns=columns,
        index=validation.index,
    )
    validate_finite_features(train_raw)
    validate_finite_features(validation_raw)
    inner_scaler = FoldStandardizer.fit(train_raw)
    train.loc[:, columns] = inner_scaler.transform(train_raw)
    validation.loc[:, columns] = inner_scaler.transform(validation_raw)
    return train, validation, train_raw, validation_raw, inner_scaler, validation_start, purged


def _load_inner_partition(
    dataset_dir: Path,
    manifest_path: Path,
    manifest: dict[str, Any],
    fold_number: int,
) -> InnerPartition:
    root = _fold_root(dataset_dir, manifest_path)
    prefix = f"fold_{fold_number:02d}"
    train_path = root / f"{prefix}_train.csv"
    scaler_path = root / f"{prefix}_scaler.json"
    feature_columns = tuple(manifest["feature_columns"])
    target_columns = tuple(manifest["target_columns"])
    try:
        outer_train = pd.read_csv(
            train_path,
            parse_dates=["signal_date", "execution_date", "next_execution_date"],
        )
        scaler_payload = json.loads(scaler_path.read_text(encoding="utf-8"))
    except (OSError, ValueError, json.JSONDecodeError) as error:
        raise MLSignalDiagnosticError(f"Cannot load 8A outer-train fold {fold_number}: {error}") from error

    if scaler_payload.get("fit_scope") != "training X rows only":
        raise MLSignalDiagnosticError(f"Fold {fold_number} outer scaler is not train-only")
    if tuple(scaler_payload.get("columns", ())) != feature_columns:
        raise MLSignalDiagnosticError(f"Fold {fold_number} scaler columns differ from the manifest")
    means = scaler_payload.get("means")
    scales = scaler_payload.get("scales")
    if not isinstance(means, dict) or not isinstance(scales, dict):
        raise MLSignalDiagnosticError(f"Fold {fold_number} scaler has no feature statistics")
    try:
        outer_scaler = FoldStandardizer(
            columns=feature_columns,
            means=tuple(float(means[column]) for column in feature_columns),
            scales=tuple(float(scales[column]) for column in feature_columns),
            constant_columns=tuple(scaler_payload.get("constant_columns", ())),
        )
    except (KeyError, TypeError, ValueError) as error:
        raise MLSignalDiagnosticError(f"Fold {fold_number} scaler is invalid: {error}") from error
    if any(not math.isfinite(value) for value in outer_scaler.means) or any(
        not math.isfinite(value) or value <= 0 for value in outer_scaler.scales
    ):
        raise MLSignalDiagnosticError(f"Fold {fold_number} scaler values are invalid")

    expected_columns = (
        "signal_date",
        "execution_date",
        "next_execution_date",
        *feature_columns,
        *target_columns,
    )
    if tuple(outer_train.columns) != expected_columns or outer_train.empty:
        raise MLSignalDiagnosticError(f"Fold {fold_number} outer-train CSV has an invalid schema")
    if not outer_train["signal_date"].is_monotonic_increasing:
        raise MLSignalDiagnosticError(f"Fold {fold_number} outer train is not chronological")
    if not np.isfinite(outer_train.loc[:, (*feature_columns, *target_columns)].to_numpy()).all():
        raise MLSignalDiagnosticError(f"Fold {fold_number} outer-train CSV is non-finite")

    (
        train,
        validation,
        train_raw,
        validation_raw,
        inner_scaler,
        validation_start,
        purged,
    ) = _split_and_scale_inner_train(outer_train, feature_columns, outer_scaler, fold_number)
    return InnerPartition(
        fold_number=fold_number,
        train=train,
        validation=validation,
        train_raw_features=train_raw,
        validation_raw_features=validation_raw,
        scaler=inner_scaler,
        validation_start=validation_start,
        purged_training_examples=purged,
    )


def _read_selected_8a_rows(
    dataset_path: Path,
    manifest: dict[str, Any],
    partition: InnerPartition,
) -> pd.DataFrame:
    selected_signals = tuple(
        sorted(
            set(partition.train["signal_date"].dt.date)
            | set(partition.validation["signal_date"].dt.date)
        )
    )
    feature_columns = tuple(manifest["feature_columns"])
    target_columns = tuple(manifest["target_columns"])
    currencies = FOREIGN_CURRENCIES
    price_columns = tuple(
        column
        for currency in currencies
        for column in (
            f"price_start_eur_per_unit_{currency}",
            f"price_end_eur_per_unit_{currency}",
        )
    )
    columns = (
        "signal_date",
        "information_cutoff_date",
        "last_feature_market_date",
        "execution_date",
        "next_execution_date",
        *feature_columns,
        *price_columns,
        *target_columns,
    )
    try:
        selected = pd.read_parquet(
            dataset_path,
            columns=list(columns),
            filters=[("signal_date", "in", list(selected_signals))],
        )
    except (OSError, ValueError) as error:
        raise MLSignalDiagnosticError(f"Cannot read selected 8A internal rows: {error}") from error
    selected["signal_date"] = pd.to_datetime(selected["signal_date"])
    selected = selected.sort_values("signal_date").reset_index(drop=True)
    if tuple(selected["signal_date"].dt.date) != selected_signals:
        raise MLSignalDiagnosticError("Filtered 8A rows do not match the requested inner signals")
    validate_8a_alignment(
        selected,
        feature_columns=feature_columns,
        target_columns=target_columns,
        development_end=date.fromisoformat(manifest["development_end"]),
        test_start=date.fromisoformat(manifest["test_start_excluded"]),
        availability_lag_days=int(manifest["availability_lag_days"]),
    )
    return selected


def _verify_fold_rows_against_8a(
    partition: InnerPartition,
    selected: pd.DataFrame,
    feature_columns: Sequence[str],
    target_columns: Sequence[str],
) -> None:
    by_signal = selected.set_index(selected["signal_date"].dt.date, drop=False)
    for label, csv_frame, restored_features in (
        ("train", partition.train, partition.train_raw_features),
        ("validation", partition.validation, partition.validation_raw_features),
    ):
        signals = tuple(csv_frame["signal_date"].dt.date)
        raw_rows = by_signal.loc[list(signals)]
        csv_features = restored_features.to_numpy(dtype=np.float64)
        parquet_features = raw_rows.loc[:, list(feature_columns)].to_numpy(dtype=np.float64)
        if not np.allclose(csv_features, parquet_features, rtol=2e-7, atol=2e-10):
            raise MLSignalDiagnosticError(
                f"Fold {partition.fold_number} {label} features differ from the immutable 8A dataset"
            )
        csv_targets = csv_frame.loc[:, list(target_columns)].to_numpy(dtype=np.float64)
        parquet_targets = raw_rows.loc[:, list(target_columns)].to_numpy(dtype=np.float64)
        if not np.allclose(csv_targets, parquet_targets, rtol=0.0, atol=1e-14):
            raise MLSignalDiagnosticError(
                f"Fold {partition.fold_number} {label} labels differ from the immutable 8A dataset"
            )


def _feature_statistics(
    features: pd.DataFrame,
    *,
    fold_number: int,
    partition_name: str,
) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for column in features.columns:
        values = pd.to_numeric(features[column], errors="coerce").to_numpy(dtype=np.float64)
        finite = np.isfinite(values)
        observed = values[finite]
        currency, indicator = _feature_parts(column)
        output.append(
            {
                "fold_number": fold_number,
                "partition": partition_name,
                "feature": column,
                "currency": currency,
                "indicator": indicator,
                "count": len(values),
                "missing_count": int(pd.isna(values).sum()),
                "nonfinite_count": int((~finite & ~pd.isna(values)).sum()),
                "finite_count": int(finite.sum()),
                "mean": float(observed.mean()) if len(observed) else None,
                "standard_deviation": float(observed.std(ddof=0)) if len(observed) else None,
                "minimum": float(observed.min()) if len(observed) else None,
                "maximum": float(observed.max()) if len(observed) else None,
                "unique_count": int(np.unique(observed).size),
                "constant": bool(len(observed) > 0 and np.ptp(observed) == 0),
            }
        )
    return output


def _feature_distribution_shifts(
    train: pd.DataFrame,
    validation: pd.DataFrame,
    scaler: FoldStandardizer,
    fold_number: int,
) -> list[dict[str, Any]]:
    train_stats = train.mean(axis=0)
    validation_stats = validation.mean(axis=0)
    output: list[dict[str, Any]] = []
    for index, column in enumerate(train.columns):
        scale = scaler.scales[index]
        output.append(
            {
                "fold_number": fold_number,
                "feature": column,
                "currency": _feature_parts(column)[0],
                "indicator": _feature_parts(column)[1],
                "train_mean": float(train_stats[column]),
                "validation_mean": float(validation_stats[column]),
                "train_standard_deviation": float(train[column].std(ddof=0)),
                "validation_standard_deviation": float(validation[column].std(ddof=0)),
                "absolute_mean_shift_in_train_sd": float(
                    abs(validation_stats[column] - train_stats[column]) / scale
                ),
                "validation_to_train_sd_ratio": float(
                    validation[column].std(ddof=0) / scale
                ),
            }
        )
    return output


def _feature_correlations(features: pd.DataFrame) -> tuple[list[list[float | None]], list[dict[str, Any]]]:
    values = features.to_numpy(dtype=np.float64)
    count = values.shape[1]
    matrix = np.full((count, count), np.nan, dtype=np.float64)
    for left in range(count):
        matrix[left, left] = 1.0 if np.std(values[:, left]) > 0 else np.nan
        for right in range(left):
            correlation = _correlation(values[:, left], values[:, right])
            matrix[left, right] = matrix[right, left] = (
                np.nan if correlation is None else correlation
            )
    pairs: list[dict[str, Any]] = []
    columns = tuple(features.columns)
    for left in range(count):
        for right in range(left):
            correlation = matrix[left, right]
            if np.isfinite(correlation):
                currency_left, indicator_left = _feature_parts(columns[left])
                currency_right, indicator_right = _feature_parts(columns[right])
                pairs.append(
                    {
                        "feature_a": columns[left],
                        "feature_b": columns[right],
                        "correlation": float(correlation),
                        "absolute_correlation": float(abs(correlation)),
                        "same_currency": currency_left == currency_right,
                        "same_indicator": indicator_left == indicator_right,
                    }
                )
    pairs.sort(key=lambda pair: pair["absolute_correlation"], reverse=True)
    serializable = [
        [float(value) if np.isfinite(value) else None for value in row]
        for row in matrix
    ]
    return serializable, pairs


def _univariate_pair(
    features: pd.DataFrame,
    targets: pd.DataFrame,
    feature: str,
    target: str,
) -> dict[str, Any]:
    x = features[feature].to_numpy(dtype=np.float64)
    y = targets[target].to_numpy(dtype=np.float64)
    pearson = _correlation(x, y)
    spearman = _spearman(x, y)
    x_low, x_high = np.quantile(x, [0.01, 0.99])
    y_low, y_high = np.quantile(y, [0.01, 0.99])
    retained = (x >= x_low) & (x <= x_high) & (y >= y_low) & (y <= y_high)
    trimmed = _correlation(x[retained], y[retained]) if retained.sum() >= 2 else None
    currency, indicator = _feature_parts(feature)
    return {
        "feature": feature,
        "currency": currency,
        "indicator": indicator,
        "target": target.removeprefix("y_simple_return_"),
        "n_observations": len(x),
        "pearson": pearson,
        "spearman": spearman,
        "direction": (
            "positive" if pearson is not None and pearson > 0
            else "negative" if pearson is not None and pearson < 0
            else "undefined_or_zero"
        ),
        "trimmed_pearson_1pct_each_tail": trimmed,
        "trimmed_observation_count": int(retained.sum()),
        "trimmed_sign_matches_full": (
            None if pearson is None or trimmed is None else bool(np.sign(pearson) == np.sign(trimmed))
        ),
    }


def _metric_bundle(
    actual: np.ndarray,
    predicted: np.ndarray,
) -> list[dict[str, Any]]:
    metrics: list[dict[str, Any]] = []
    for index, currency in enumerate(FOREIGN_CURRENCIES):
        y = actual[:, index]
        estimate = predicted[:, index]
        residual = estimate - y
        total_variation = float(np.square(y - y.mean()).sum())
        metrics.append(
            {
                "currency": currency,
                "n_observations": len(y),
                "mse": float(np.square(residual).mean()),
                "mae": float(np.abs(residual).mean()),
                "r2": (
                    float(1.0 - np.square(residual).sum() / total_variation)
                    if total_variation > 0
                    else None
                ),
                "prediction_realized_correlation": _correlation(estimate, y),
                "directional_accuracy": float((np.sign(estimate) == np.sign(y)).mean()),
            }
        )
    return metrics


def _cross_sectional_rank(actual: np.ndarray, predicted: np.ndarray) -> dict[str, Any]:
    values: list[float] = []
    for row in range(len(actual)):
        correlation = _spearman(predicted[row], actual[row])
        if correlation is not None:
            values.append(correlation)
    return {
        "mean_weekly_rank_correlation": float(np.mean(values)) if values else None,
        "median_weekly_rank_correlation": float(np.median(values)) if values else None,
        "defined_weeks": len(values),
        "total_weeks": len(actual),
    }


def _mean_prediction(targets: np.ndarray) -> np.ndarray:
    if targets.ndim != 2 or len(targets) == 0 or not np.isfinite(targets).all():
        raise MLSignalDiagnosticError("Constant predictor requires a finite non-empty target matrix")
    return np.broadcast_to(targets.mean(axis=0), targets.shape).copy()


def _grouped_feature_profiles(
    profiles: Sequence[dict[str, Any]],
    shifts: Sequence[dict[str, Any]],
) -> dict[str, list[dict[str, Any]]]:
    train_rows = [
        item for item in profiles
        if item["fold_number"] == 3 and item["partition"] == "inner_train"
    ]
    val_rows = [
        item for item in profiles
        if item["fold_number"] == 3 and item["partition"] == "inner_validation"
    ]
    shift_by_key = {
        (item["fold_number"], item["feature"]): item
        for item in shifts
        if item["fold_number"] == 3
    }
    groups: dict[str, list[dict[str, Any]]] = {"by_indicator": [], "by_currency": []}
    for group_name, key_name in (("by_indicator", "indicator"), ("by_currency", "currency")):
        keys = sorted({item[key_name] for item in train_rows})
        for key in keys:
            train_group = [item for item in train_rows if item[key_name] == key]
            val_group = [item for item in val_rows if item[key_name] == key]
            shift_values = [
                shift_by_key[(shift["fold_number"], shift["feature"])]["absolute_mean_shift_in_train_sd"]
                for shift in shifts
                if shift["fold_number"] == 3 and shift[key_name] == key
            ]
            train_sds = [
                item["standard_deviation"]
                for item in train_group
                if item["standard_deviation"] is not None
            ]
            groups[group_name].append(
                {
                    "group": key,
                    "feature_count": len(train_group),
                    "missing_count": sum(item["missing_count"] for item in train_group),
                    "nonfinite_count": sum(item["nonfinite_count"] for item in train_group),
                    "constant_feature_partition_count": sum(
                        item["constant"] for item in train_group
                    ),
                    "median_feature_standard_deviation": (
                        float(np.median(train_sds)) if train_sds else None
                    ),
                    "median_absolute_train_validation_mean_shift_in_train_sd": (
                        float(np.median(shift_values)) if shift_values else None
                    ),
                    "validation_feature_count": len(val_group),
                }
            )
    return groups


def _plot_feature_correlation(
    matrix: Sequence[Sequence[float | None]],
    features: Sequence[str],
    path: Path,
) -> None:
    values = np.asarray(
        [[np.nan if value is None else value for value in row] for row in matrix],
        dtype=np.float64,
    )
    figure, axis = plt.subplots(figsize=(19, 17))
    image = axis.imshow(np.ma.masked_invalid(values), cmap="RdBu_r", vmin=-1, vmax=1)
    labels = [feature.replace("__", "\n") for feature in features]
    axis.set_xticks(np.arange(len(labels)), labels=labels, rotation=90, fontsize=5)
    axis.set_yticks(np.arange(len(labels)), labels=labels, fontsize=5)
    for boundary in range(13, len(labels), 13):
        axis.axhline(boundary - 0.5, color="black", linewidth=0.5, alpha=0.45)
        axis.axvline(boundary - 0.5, color="black", linewidth=0.5, alpha=0.45)
    axis.set_title("Corrélations Pearson entre features — inner train du fold 3")
    figure.colorbar(image, ax=axis, shrink=0.75, label="Pearson r")
    figure.tight_layout()
    figure.savefig(path, dpi=150)
    plt.close(figure)


def _plot_r2(
    folds: Sequence[dict[str, Any]],
    pooled: dict[str, Any],
    path: Path,
) -> None:
    figure, axes = plt.subplots(2, 2, figsize=(14, 9), sharey=True)
    entries = [(item["fold_number"], item["model_metrics"]) for item in folds]
    entries.append(("Pooled", pooled["model_metrics"]))
    positions = np.arange(len(FOREIGN_CURRENCIES))
    width = 0.36
    for axis, (label, metrics) in zip(axes.flat, entries, strict=True):
        for offset, model_name in enumerate(MODEL_NAMES):
            by_currency = {item["currency"]: item for item in metrics[model_name]}
            values = [
                np.nan if by_currency[currency]["r2"] is None else by_currency[currency]["r2"]
                for currency in FOREIGN_CURRENCIES
            ]
            axis.bar(
                positions + (offset - 0.5) * width,
                values,
                width,
                label="Prédiction moyenne train" if model_name == "constant_mean" else "Ridge α=10",
            )
        axis.axhline(0, color="black", linewidth=0.7)
        axis.set_title(f"Fold {label}" if isinstance(label, int) else str(label))
        axis.set_xticks(positions, labels=FOREIGN_CURRENCIES)
        axis.grid(True, axis="y", alpha=0.25)
    axes[0, 0].set_ylabel("R² hors entraînement")
    axes[1, 0].set_ylabel("R² hors entraînement")
    axes[1, 0].legend()
    figure.suptitle("Pouvoir prédictif par devise et segment interne")
    figure.tight_layout()
    figure.savefig(path, dpi=150)
    plt.close(figure)


def _format(value: Any, digits: int = 4) -> str:
    return "N/D" if value is None else f"{value:.{digits}f}"


def _json_default(value: Any) -> Any:
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value) if np.isfinite(value) else None
    if isinstance(value, np.bool_):
        return bool(value)
    raise TypeError(f"Unsupported JSON value {type(value).__name__}")


def _render_report(payload: dict[str, Any]) -> str:
    lines = [
        "# Diagnostic du signal prédictif des features Forex — étape 8E",
        "",
        "## Périmètre et méthode",
        "",
        f"- Dataset 8A immuable : `{payload['dataset_content_sha256']}`; {payload['dataset_example_count']} exemples validés.",
        f"- Développement limité au {payload['development_end']}; test final à partir du {payload['test_start_excluded']} exclu.",
        "- 91 features, 13 indicateurs × 7 devises; targets = rendements simples EUR/devise entre les exécutions 8A.",
        "- Les trois splits sont les validations internes 8C, sur les seuls outer-trains, avec purge stricte et scaler ajusté sur l'inner train.",
        f"- Ridge multivarié : alpha fixé à {payload['ridge_alpha']:.1f}, features standardisées train-only, aucune recherche; prédicteur constant = moyennes train par devise.",
        "- Les validations externes ne sont ni lues ni utilisées. Les périodes internes ont déjà contribué aux choix 8B–8D : résultats exploratoires, non indépendants.",
        "",
        "## Comparaison prédictive Ridge vs constante",
        "",
        "R² est le R² hors entraînement usuel, calculé par devise autour de la moyenne réalisée dans la période de validation. Il peut être négatif. Les métriques sont des erreurs de rendement, pas des performances de stratégie.",
        "",
        "| Période | Modèle | MSE moyen (7 devises) | MAE moyen | R² moyen | Corr. prédit/réalisé moyenne | Direction correcte moyenne | IC rang transversal hebdo moyen | Semaines IC définies |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for period, metrics in [
        *((str(item["fold_number"]), item["model_metrics"]) for item in payload["folds"]),
        ("Pooled", payload["pooled"]["model_metrics"]),
    ]:
        for model_name in MODEL_NAMES:
            currency_metrics = metrics[model_name]
            correlations = [
                item["prediction_realized_correlation"]
                for item in currency_metrics
                if item["prediction_realized_correlation"] is not None
            ]
            r2s = [item["r2"] for item in currency_metrics if item["r2"] is not None]
            direction = [item["directional_accuracy"] for item in currency_metrics]
            suffix = payload["pooled"] if period == "Pooled" else next(
                item for item in payload["folds"] if str(item["fold_number"]) == period
            )
            rank = suffix["cross_sectional_rank"][model_name]
            lines.append(
                f"| {period} | {'Moyenne train' if model_name == 'constant_mean' else 'Ridge α=10'} | "
                f"{np.mean([item['mse'] for item in currency_metrics]):.8g} | "
                f"{np.mean([item['mae'] for item in currency_metrics]):.8g} | "
                f"{_format(float(np.mean(r2s)) if r2s else None)} | "
                f"{_format(float(np.mean(correlations)) if correlations else None)} | "
                f"{np.mean(direction):.3f} | "
                f"{_format(rank['mean_weekly_rank_correlation'])} | "
                f"{rank['defined_weeks']}/{rank['total_weeks']} |"
            )

    lines.extend(
        [
            "",
            "### Métriques par devise et segment",
            "",
            "| Période | Modèle | Devise | n | MSE | MAE | R² | Corrélation | Direction |",
            "|---|---|---|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for period, metrics in [
        *((str(item["fold_number"]), item["model_metrics"]) for item in payload["folds"]),
        ("Pooled", payload["pooled"]["model_metrics"]),
    ]:
        for model_name in MODEL_NAMES:
            for item in metrics[model_name]:
                lines.append(
                    f"| {period} | {'Moyenne train' if model_name == 'constant_mean' else 'Ridge'} | "
                    f"{item['currency']} | {item['n_observations']} | {item['mse']:.8g} | "
                    f"{item['mae']:.8g} | {_format(item['r2'])} | "
                    f"{_format(item['prediction_realized_correlation'])} | "
                    f"{item['directional_accuracy']:.3f} |"
                )

    lines.extend(
        [
            "",
            "## Corrélations univariées X_t → rendement futur Y_t",
            "",
            "Chaque feature est reliée uniquement au rendement futur de sa propre devise. Les trois colonnes de validation interne sont présentées séparément; pooled agrège ces segments disjoints à titre descriptif. Aucune feature n'est choisie. Les corrélations ne sont pas des preuves de prédictivité, particulièrement avec 91 comparaisons corrélées et de petits segments.",
            "",
            "| Devise | Indicateur | n (fold 1/2/3) | Pearson pooled | Spearman pooled | Pearson fold 1 | fold 2 | fold 3 | Signe stable (n/3) | Pearson après exclusion des 1% extrêmes |",
            "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for item in payload["univariate_correlations"]:
        fold_corr = item["fold_pearson"]
        lines.append(
            f"| {item['currency']} | {item['indicator']} | "
            f"{'/'.join(str(value) for value in item['fold_observation_counts'])} | "
            f"{_format(item['pearson'])} | {_format(item['spearman'])} | "
            f"{_format(fold_corr[0])} | {_format(fold_corr[1])} | {_format(fold_corr[2])} | "
            f"{item['same_sign_fold_count']}/3 | "
            f"{_format(item['trimmed_pearson_1pct_each_tail'])} |"
        )

    lines.extend(
        [
            "",
            "## Qualité, distributions et redondance des features",
            "",
            "Le profil détaillé (moyenne, écart-type population, min/max, cardinalité, valeurs manquantes/non finies et constance) pour chaque feature et chacun des trains/validations internes est inclus dans le JSON. Les résumés groupés ci-dessous utilisent le plus grand segment d'entraînement interne, fold 3; les décalages distributionnels sont calculés fold par fold à l'échelle de l'écart-type train, sans fitter sur validation.",
            "",
            "| Indicateur | Features | Constantes | Manquantes | Non finies | Médiane des écarts-types bruts | Médiane décalage moyenne val/train (écarts-types train) |",
            "|---|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for item in payload["feature_group_profiles"]["by_indicator"]:
        lines.append(
            f"| {item['group']} | {item['feature_count']} | "
            f"{item['constant_feature_partition_count']} | {item['missing_count']} | "
            f"{item['nonfinite_count']} | {_format(item['median_feature_standard_deviation'])} | "
            f"{_format(item['median_absolute_train_validation_mean_shift_in_train_sd'])} |"
        )
    lines.extend(
        [
            "",
            "| Devise | Features | Constantes | Manquantes | Non finies | Médiane des écarts-types bruts | Médiane décalage moyenne val/train (écarts-types train) |",
            "|---|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for item in payload["feature_group_profiles"]["by_currency"]:
        lines.append(
            f"| {item['group']} | {item['feature_count']} | "
            f"{item['constant_feature_partition_count']} | {item['missing_count']} | "
            f"{item['nonfinite_count']} | {_format(item['median_feature_standard_deviation'])} | "
            f"{_format(item['median_absolute_train_validation_mean_shift_in_train_sd'])} |"
        )
    high = payload["high_feature_correlations"]
    lines.extend(
        [
            "",
            (
                f"La carte complète fold 3 est disponible dans "
                f"`./{payload['charts']['feature_correlations']}`. {len(high)} paires ont "
                "|Pearson r| ≥ 0,95 dans le train interne fold 3; ces paires ne sont pas supprimées."
            ),
            "",
            "| Feature A | Feature B | Pearson r | Même devise | Même indicateur |",
            "|---|---|---:|---|---|",
        ]
    )
    for item in high[:30]:
        lines.append(
            f"| {item['feature_a']} | {item['feature_b']} | {item['correlation']:.4f} | "
            f"{'oui' if item['same_currency'] else 'non'} | "
            f"{'oui' if item['same_indicator'] else 'non'} |"
        )
    lines.extend(
        [
            "",
            "La matrice numérique complète est conservée dans le JSON. La redondance est descriptive; aucune sélection, combinaison ou suppression n'est faite.",
            "",
            "## Stabilité temporelle et sensibilité aux extrêmes",
            "",
            (
                f"- Signes identiques sur les trois fenêtres pour toutes les corrélations définies : "
                f"{payload['stability']['univariate_same_sign_all_three']}/"
                f"{payload['stability']['univariate_defined_all_three']} paires."
            ),
            (
                f"- Paires dont le signe change après exclusion des 1% de queues X/Y : "
                f"{payload['stability']['trimmed_sign_changes']}/"
                f"{payload['stability']['trimmed_defined']} corrélations pooled définies."
            ),
            (
                f"- Réduction MSE Ridge vs prédicteur constant, moyenne des 7 devises en pooled : "
                f"{_format(payload['stability']['ridge_mse_relative_improvement'])}. "
                "Valeur positive = MSE plus faible pour Ridge; valeur négative = dégradation."
            ),
            "- Les valeurs absentes (corrélation/R² indéfini) sont `null` dans le JSON et N/D dans le rapport; elles ne sont pas remplacées par zéro.",
            "",
            "## Conclusion de l'audit",
            "",
            (
                f"1. **Pouvoir prédictif mesurable :** Ridge alpha=10 "
                f"{'réduit' if (payload['stability']['ridge_mse_relative_improvement'] or 0) > 0 else 'ne réduit pas'} "
                "le MSE moyen pooled par rapport à la moyenne train; ce diagnostic descriptif ne "
                "constitue pas un test statistique de signal exploitable."
            ),
            (
                f"2. **Stabilité :** {payload['stability']['univariate_same_sign_all_three']} "
                "corrélations univariées sur "
                f"{payload['stability']['univariate_defined_all_three']} gardent leur signe dans "
                "les trois fenêtres; le faible nombre d'observations rend les estimations incertaines."
            ),
            (
                f"3. **Modèle linéaire :** Ridge "
                f"{'fait mieux' if (payload['stability']['ridge_mse_relative_improvement'] or 0) > 0 else 'ne fait pas mieux'} "
                "en MSE pooled que le prédicteur constant. Les résultats par devise/période et R² "
                "négatifs restent visibles ci-dessus."
            ),
            "4. **Interprétation du MLP :** un signal faible ou instable est une explication compatible avec les résultats, mais cet audit ne prouve pas que l'architecture est adéquate ou inadéquate.",
            "5. **Hypothèses futures, non implémentées :** signal différent selon les régimes; cible dominée par le bruit des fixings hebdomadaires; relation non linéaire; capacité de prévision du signe distincte de la taille des rendements; redondance inter-devises/indicateurs; limites de disponibilité point-in-time FRED. Toute étude ultérieure devra être préspécifiée et utiliser un test encore vierge.",
            "",
            "## Limites statistiques",
            "",
            "- Chaque validation interne ne contient qu'une fraction du petit historique de développement; les folds ont été consultés dans les étapes précédentes.",
            "- 91 corrélations univariées (et des features fortement corrélées) créent un risque de découvertes fortuites; aucune p-value non corrigée n'est présentée comme preuve.",
            "- Les valeurs FRED sont indicatives et leur heure de publication effective n'est pas attestée; le délai 7 jours reste une convention.",
            "- Les rendements sont des prix de référence, sans frais, spread, slippage, intérêts ou carry; aucune allocation/trading Ridge n'est simulée.",
            "",
            "## Graphiques",
            "",
            f"- Carte de corrélation des features : `./{payload['charts']['feature_correlations']}`",
            f"- R² par devise, modèle et fenêtre : `./{payload['charts']['r2']}`",
            "",
        ]
    )
    return "\n".join(lines)


def run_diagnostic(
    *,
    dataset_dir: Path | None = None,
    report_path: Path | None = None,
) -> dict[str, Any]:
    """Run the fixed-alpha diagnostic without touching 8A data or loading outer validation."""
    resolved_dataset_dir = dataset_dir or PROJECT_ROOT / "data" / "derived" / "ml"
    resolved_report_path = report_path or PROJECT_ROOT / "reports" / "ml_signal_diagnostic.md"
    dataset_path, manifest_path, manifest = _load_manifest(resolved_dataset_dir)
    dataset_hash_before = sha256_file(dataset_path)
    parquet_metadata = pq.read_metadata(dataset_path).metadata or {}
    dataset_content_sha256 = parquet_metadata.get(
        b"forex_v1.ml_dataset.content_sha256", b""
    ).decode("ascii")
    if dataset_content_sha256 != manifest.get("content_sha256"):
        raise MLSignalDiagnosticError("8A Parquet content hash metadata differs from its manifest")

    feature_columns = tuple(manifest["feature_columns"])
    target_columns = _target_columns(manifest["target_columns"])
    partitions: list[InnerPartition] = []
    aligned_rows: list[pd.DataFrame] = []
    for fold_number in range(1, int(manifest["walk_forward"]["fold_count"]) + 1):
        partition = _load_inner_partition(
            resolved_dataset_dir,
            manifest_path,
            manifest,
            fold_number,
        )
        selected = _read_selected_8a_rows(dataset_path, manifest, partition)
        _verify_fold_rows_against_8a(
            partition,
            selected,
            feature_columns,
            manifest["target_columns"],
        )
        partitions.append(partition)
        aligned_rows.append(selected)

    all_validation_signals = [
        signal
        for partition in partitions
        for signal in partition.validation["signal_date"].dt.date
    ]
    if len(all_validation_signals) != len(set(all_validation_signals)):
        raise MLSignalDiagnosticError("Internal validation windows overlap")

    feature_profiles: list[dict[str, Any]] = []
    distribution_shifts: list[dict[str, Any]] = []
    fold_results: list[dict[str, Any]] = []
    prediction_store: dict[str, list[np.ndarray]] = {name: [] for name in MODEL_NAMES}
    actual_store: list[np.ndarray] = []
    univariate_by_fold: dict[str, dict[str, dict[str, Any]]] = {}
    pooled_features: list[pd.DataFrame] = []
    pooled_targets: list[pd.DataFrame] = []
    corr_matrix: list[list[float | None]] | None = None
    feature_corr_pairs: list[dict[str, Any]] = []
    charts_dir = resolved_report_path.parent
    charts_dir.mkdir(parents=True, exist_ok=True)

    for partition, selected in zip(partitions, aligned_rows, strict=True):
        raw_by_signal = selected.set_index(selected["signal_date"].dt.date, drop=False)
        train_rows = raw_by_signal.loc[
            list(partition.train["signal_date"].dt.date)
        ].reset_index(drop=True)
        validation_rows = raw_by_signal.loc[
            list(partition.validation["signal_date"].dt.date)
        ].reset_index(drop=True)
        train_features = train_rows.loc[:, list(feature_columns)]
        validation_features = validation_rows.loc[:, list(feature_columns)]
        train_targets = train_rows.loc[:, list(target_columns)]
        validation_targets = validation_rows.loc[:, list(target_columns)]
        x_train = partition.scaler.transform(train_features).to_numpy(dtype=np.float64)
        x_validation = partition.scaler.transform(validation_features).to_numpy(dtype=np.float64)
        y_train = train_targets.to_numpy(dtype=np.float64)
        y_validation = validation_targets.to_numpy(dtype=np.float64)
        validate_finite_features(train_features)
        validate_finite_features(validation_features)
        feature_profiles.extend(
            _feature_statistics(
                train_features,
                fold_number=partition.fold_number,
                partition_name="inner_train",
            )
        )
        feature_profiles.extend(
            _feature_statistics(
                validation_features,
                fold_number=partition.fold_number,
                partition_name="inner_validation",
            )
        )
        distribution_shifts.extend(
            _feature_distribution_shifts(
                train_features,
                validation_features,
                partition.scaler,
                partition.fold_number,
            )
        )

        ridge_coef, ridge_intercept = fit_ridge(x_train, y_train, RIDGE_ALPHA)
        predictions = {
            "constant_mean": np.broadcast_to(y_train.mean(axis=0), y_validation.shape).copy(),
            "ridge": predict_ridge(x_validation, ridge_coef, ridge_intercept),
        }
        actual_store.append(y_validation)
        for model_name, values in predictions.items():
            prediction_store[model_name].append(values)
        fold_targets = {
            "constant_mean": _metric_bundle(y_validation, predictions["constant_mean"]),
            "ridge": _metric_bundle(y_validation, predictions["ridge"]),
        }
        fold_results.append(
            {
                "fold_number": partition.fold_number,
                "inner_train_count": len(partition.train),
                "inner_validation_count": len(partition.validation),
                "purged_training_examples": partition.purged_training_examples,
                "validation_start": partition.validation_start.isoformat(),
                "inner_train_first_signal": partition.train["signal_date"].min().date().isoformat(),
                "inner_train_last_signal": partition.train["signal_date"].max().date().isoformat(),
                "inner_train_last_label_end": partition.train["next_execution_date"].max().date().isoformat(),
                "inner_validation_last_signal": partition.validation["signal_date"].max().date().isoformat(),
                "inner_validation_last_label_end": partition.validation["next_execution_date"].max().date().isoformat(),
                "scaler_fit_scope": "inner train rows only",
                "scaler_means": dict(zip(feature_columns, partition.scaler.means, strict=True)),
                "scaler_scales": dict(zip(feature_columns, partition.scaler.scales, strict=True)),
                "constant_features": list(partition.scaler.constant_columns),
                "model_metrics": fold_targets,
                "cross_sectional_rank": {
                    name: _cross_sectional_rank(y_validation, values)
                    for name, values in predictions.items()
                },
            }
        )
        fold_id = str(partition.fold_number)
        univariate_by_fold[fold_id] = {}
        for feature in feature_columns:
            currency, _ = _feature_parts(feature)
            target = f"y_simple_return_{currency}"
            univariate_by_fold[fold_id][feature] = _univariate_pair(
                validation_features,
                validation_targets,
                feature,
                target,
            )
        pooled_features.append(validation_features.reset_index(drop=True))
        pooled_targets.append(validation_targets.reset_index(drop=True))
        if partition.fold_number == 3:
            corr_matrix, feature_corr_pairs = _feature_correlations(train_features)

    if corr_matrix is None:
        raise MLSignalDiagnosticError("Fold 3 internal training data were not analyzed")
    pooled_x = pd.concat(pooled_features, ignore_index=True)
    pooled_y = pd.concat(pooled_targets, ignore_index=True)
    pooled_univariate: list[dict[str, Any]] = []
    for feature in feature_columns:
        currency, _ = _feature_parts(feature)
        pooled_univariate.append(
            _univariate_pair(pooled_x, pooled_y, feature, f"y_simple_return_{currency}")
        )
    for result in pooled_univariate:
        fold_corr = [
            univariate_by_fold[str(number)][result["feature"]]["pearson"]
            for number in (1, 2, 3)
        ]
        fold_counts = [
            univariate_by_fold[str(number)][result["feature"]]["n_observations"]
            for number in (1, 2, 3)
        ]
        defined = [value for value in fold_corr if value is not None and value != 0]
        result["fold_pearson"] = fold_corr
        result["fold_observation_counts"] = fold_counts
        result["same_sign_fold_count"] = (
            sum(np.sign(value) == np.sign(defined[0]) for value in defined)
            if defined
            else 0
        )
        result["defined_fold_count"] = len(defined)
        result["fold_details"] = [
            univariate_by_fold[str(number)][result["feature"]]
            for number in (1, 2, 3)
        ]

    pooled_actual = np.concatenate(actual_store, axis=0)
    pooled_predictions = {
        name: np.concatenate(values, axis=0)
        for name, values in prediction_store.items()
    }
    pooled_metrics = {
        name: _metric_bundle(pooled_actual, values)
        for name, values in pooled_predictions.items()
    }
    pooled_rank = {
        name: _cross_sectional_rank(pooled_actual, values)
        for name, values in pooled_predictions.items()
    }
    baseline_mse = float(np.mean([item["mse"] for item in pooled_metrics["constant_mean"]]))
    ridge_mse = float(np.mean([item["mse"] for item in pooled_metrics["ridge"]]))
    relative_improvement = (
        1.0 - ridge_mse / baseline_mse if baseline_mse > 0 else None
    )
    all_three = [item for item in pooled_univariate if item["defined_fold_count"] == 3]
    same_sign_all_three = [
        item
        for item in all_three
        if len({int(np.sign(value)) for value in item["fold_pearson"] if value is not None}) == 1
    ]
    trimmed_defined = [
        item for item in pooled_univariate if item["pearson"] is not None
    ]
    trimmed_sign_changes = sum(
        item["trimmed_sign_matches_full"] is False for item in trimmed_defined
    )
    groups = _grouped_feature_profiles(feature_profiles, distribution_shifts)
    correlation_chart = charts_dir / "ml_signal_feature_correlation.png"
    r2_chart = charts_dir / "ml_signal_r2.png"
    _plot_feature_correlation(corr_matrix, feature_columns, correlation_chart)
    pooled_report_metrics = {
        "model_metrics": pooled_metrics,
        "cross_sectional_rank": pooled_rank,
    }
    _plot_r2(fold_results, pooled_report_metrics, r2_chart)

    dataset_hash_after = sha256_file(dataset_path)
    if dataset_hash_after != dataset_hash_before:
        raise MLSignalDiagnosticError("The immutable 8A Parquet changed during the diagnostic")
    payload: dict[str, Any] = {
        "diagnostic": "step_8e_ml_feature_signal",
        "dataset_content_sha256": dataset_content_sha256,
        "dataset_file_sha256_before": dataset_hash_before,
        "dataset_file_sha256_after": dataset_hash_after,
        "dataset_example_count": int(manifest["example_count"]),
        "dataset_selected_internal_example_count": len({
            signal for item in partitions
            for signal in (
                list(item.train["signal_date"].dt.date)
                + list(item.validation["signal_date"].dt.date)
            )
        }),
        "development_end": manifest["development_end"],
        "test_start_excluded": manifest["test_start_excluded"],
        "test_period_evaluated": False,
        "outer_validation_loaded": False,
        "feature_count": len(feature_columns),
        "indicator_count": len(manifest["feature_names"]),
        "currency_count": len(FOREIGN_CURRENCIES),
        "feature_columns": list(feature_columns),
        "indicator_names": list(manifest["feature_names"]),
        "target_currencies": list(FOREIGN_CURRENCIES),
        "availability_lag_days": int(manifest["availability_lag_days"]),
        "inner_validation_fraction": INNER_VALIDATION_FRACTION,
        "ridge_alpha": RIDGE_ALPHA,
        "ridge_objective": "sum squared errors + alpha * squared L2 coefficient norm, with intercept",
        "constant_predictor": "per-foreign-currency mean target fitted on inner train",
        "feature_statistics": feature_profiles,
        "feature_distribution_shifts": distribution_shifts,
        "feature_group_profiles": groups,
        "feature_correlation_fold_3_inner_train": {
            "method": "Pearson",
            "features": list(feature_columns),
            "matrix": corr_matrix,
        },
        "high_feature_correlations": [
            item for item in feature_corr_pairs if item["absolute_correlation"] >= 0.95
        ],
        "top_feature_correlations": feature_corr_pairs[:100],
        "folds": fold_results,
        "univariate_correlations": pooled_univariate,
        "pooled": {
            "model_metrics": pooled_metrics,
            "cross_sectional_rank": pooled_rank,
        },
        "stability": {
            "univariate_same_sign_all_three": len(same_sign_all_three),
            "univariate_defined_all_three": len(all_three),
            "trimmed_sign_changes": int(trimmed_sign_changes),
            "trimmed_defined": len(trimmed_defined),
            "ridge_mse_relative_improvement": relative_improvement,
        },
        "charts": {
            "feature_correlations": correlation_chart.name,
            "r2": r2_chart.name,
        },
    }
    report_text = _render_report(payload)
    resolved_report_path.parent.mkdir(parents=True, exist_ok=True)
    resolved_report_path.write_text(report_text, encoding="utf-8")
    resolved_report_path.with_suffix(".json").write_text(
        json.dumps(
            payload,
            indent=2,
            sort_keys=True,
            allow_nan=False,
            default=_json_default,
        )
        + "\n",
        encoding="utf-8",
    )
    return payload


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Audit 8A Forex feature distributions and one-week predictive signal."
    )
    parser.add_argument("--dataset-dir", type=Path, default=PROJECT_ROOT / "data" / "derived" / "ml")
    parser.add_argument(
        "--report",
        type=Path,
        default=PROJECT_ROOT / "reports" / "ml_signal_diagnostic.md",
    )
    arguments = parser.parse_args(argv)
    result = run_diagnostic(dataset_dir=arguments.dataset_dir, report_path=arguments.report)
    print(f"Signal diagnostic report: {arguments.report}")
    print(f"Signal diagnostic metrics: {arguments.report.with_suffix('.json')}")
    print(f"Ridge alpha: {result['ridge_alpha']}; test period evaluated: {result['test_period_evaluated']}")
    for fold in result["folds"]:
        ridge = fold["model_metrics"]["ridge"]
        mean_r2 = np.mean([item["r2"] for item in ridge if item["r2"] is not None])
        print(
            f"Fold {fold['fold_number']}: train={fold['inner_train_count']}, "
            f"validation={fold['inner_validation_count']}, ridge mean R2={mean_r2:.4f}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
