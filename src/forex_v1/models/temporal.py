"""Purged walk-forward folds and fold-scoped feature standardization."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, timedelta
from typing import Any

import numpy as np
import pandas as pd


@dataclass(frozen=True, slots=True)
class WalkForwardFold:
    """Indices and chronological boundaries for one purged validation fold."""

    fold_number: int
    train_indices: tuple[int, ...]
    validation_indices: tuple[int, ...]
    training_start: date
    training_end_signal: date
    validation_start: date
    validation_end_signal: date
    purged_training_examples: int
    embargo_days: int


def _as_date(value: Any) -> date:
    if isinstance(value, pd.Timestamp):
        return value.date()
    if isinstance(value, date):
        return value
    return pd.Timestamp(value).date()


def _add_years(value: date, years: int) -> date:
    try:
        return value.replace(year=value.year + years)
    except ValueError:
        return value.replace(year=value.year + years, day=28)


def build_walk_forward_folds(
    rows: tuple[dict[str, Any], ...] | list[dict[str, Any]],
    *,
    fold_count: int,
    minimum_training_years: int,
    embargo_days: int = 0,
) -> tuple[WalkForwardFold, ...]:
    """Create expanding windows and purge training labels overlapping validation."""
    if not rows:
        raise ValueError("rows must not be empty")
    if (
        isinstance(fold_count, bool)
        or not isinstance(fold_count, int)
        or fold_count < 2
    ):
        raise ValueError("fold_count must be an integer of at least 2")
    if (
        isinstance(minimum_training_years, bool)
        or not isinstance(minimum_training_years, int)
        or minimum_training_years < 1
    ):
        raise ValueError("minimum_training_years must be a positive integer")
    if isinstance(embargo_days, bool) or not isinstance(embargo_days, int) or embargo_days < 0:
        raise ValueError("embargo_days must be a non-negative integer")

    signals = [_as_date(row["signal_date"]) for row in rows]
    endpoints = [_as_date(row["next_execution_date"]) for row in rows]
    if signals != sorted(signals) or len(set(signals)) != len(signals):
        raise ValueError("rows must be ordered by unique signal_date")
    first_validation_eligible = _add_years(signals[0], minimum_training_years)
    first_validation_index = next(
        (index for index, signal in enumerate(signals) if signal >= first_validation_eligible),
        len(signals),
    )
    remaining = len(signals) - first_validation_index
    if remaining < fold_count:
        raise ValueError(
            "Not enough examples after the minimum training period for the requested folds"
        )
    validation_chunks = [
        chunk.tolist()
        for chunk in np.array_split(np.arange(first_validation_index, len(signals)), fold_count)
    ]
    folds: list[WalkForwardFold] = []
    for fold_number, validation_indices_list in enumerate(validation_chunks, start=1):
        validation_indices = tuple(int(index) for index in validation_indices_list)
        validation_start = signals[validation_indices[0]]
        validation_end = signals[validation_indices[-1]]
        eligible_training_indices = [
            index for index, signal in enumerate(signals) if signal < validation_start
        ]
        train_indices = tuple(
            index
            for index in eligible_training_indices
            if endpoints[index] + timedelta(days=embargo_days) < validation_start
        )
        purged = len(eligible_training_indices) - len(train_indices)
        if not train_indices:
            raise ValueError(f"Fold {fold_number} has no training examples after purging")
        if (validation_start - signals[train_indices[0]]).days < 365 * minimum_training_years:
            raise ValueError(f"Fold {fold_number} does not meet the minimum training span")
        folds.append(
            WalkForwardFold(
                fold_number=fold_number,
                train_indices=train_indices,
                validation_indices=validation_indices,
                training_start=signals[train_indices[0]],
                training_end_signal=signals[train_indices[-1]],
                validation_start=validation_start,
                validation_end_signal=validation_end,
                purged_training_examples=purged,
                embargo_days=embargo_days,
            )
        )
    return tuple(folds)


@dataclass(frozen=True, slots=True)
class FoldStandardizer:
    """Per-column population standardizer fitted on one training partition only."""

    columns: tuple[str, ...]
    means: tuple[float, ...]
    scales: tuple[float, ...]
    constant_columns: tuple[str, ...]

    @classmethod
    def fit(cls, training: pd.DataFrame) -> FoldStandardizer:
        if training.empty:
            raise ValueError("training features must not be empty")
        if not training.columns.is_unique:
            raise ValueError("feature columns must be unique")
        values = training.to_numpy(dtype=float)
        if not np.isfinite(values).all():
            raise ValueError("training features must be finite; imputation is not implicit")
        means_array = values.mean(axis=0)
        raw_scales = values.std(axis=0, ddof=0)
        constant = raw_scales == 0
        scales_array = np.where(constant, 1.0, raw_scales)
        return cls(
            columns=tuple(str(column) for column in training.columns),
            means=tuple(float(value) for value in means_array),
            scales=tuple(float(value) for value in scales_array),
            constant_columns=tuple(
                str(training.columns[index]) for index, is_constant in enumerate(constant) if is_constant
            ),
        )

    def transform(self, features: pd.DataFrame) -> pd.DataFrame:
        if tuple(str(column) for column in features.columns) != self.columns:
            raise ValueError("feature columns and order must match the fitted scaler")
        values = features.to_numpy(dtype=float)
        if not np.isfinite(values).all():
            raise ValueError("features to transform must be finite")
        transformed = (values - np.asarray(self.means)) / np.asarray(self.scales)
        if not np.isfinite(transformed).all():
            raise ValueError("standardized features are not finite")
        return pd.DataFrame(transformed, columns=self.columns, index=features.index)

    def to_dict(self) -> dict[str, Any]:
        return {
            "method": "standard score per currency-feature column",
            "variance_convention": "population standard deviation (ddof=0)",
            "columns": list(self.columns),
            "means": dict(zip(self.columns, self.means, strict=True)),
            "scales": dict(zip(self.columns, self.scales, strict=True)),
            "constant_columns": list(self.constant_columns),
            "constant_column_scale": 1.0,
        }
