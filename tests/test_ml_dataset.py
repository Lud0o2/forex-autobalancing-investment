from __future__ import annotations

import hashlib
import json
from datetime import date, timedelta

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import pytest

from forex_v1.backtest.data import MarketData
from forex_v1.models.dataset import (
    MLDatasetConfig,
    MLDatasetError,
    build_ml_dataset,
    export_ml_dataset,
)
from forex_v1.models.temporal import FoldStandardizer, build_walk_forward_folds


def synthetic_market_data(
    *,
    start: date = date(2000, 1, 3),
    end: date = date(2012, 12, 31),
    skip: set[date] | None = None,
    change_prices_after: date | None = None,
    change_features_after: date | None = None,
) -> MarketData:
    skipped = skip or set()
    prices: dict[date, dict[str, float]] = {}
    features: dict[date, dict[str, float]] = {}
    current = start
    index = 0
    currencies = ("USD", "GBP", "JPY", "CHF", "CAD", "AUD", "NZD")
    while current <= end:
        if current.weekday() < 5 and current not in skipped:
            shifted = change_prices_after is not None and current > change_prices_after
            price_multiplier = 5.0 if shifted else 1.0
            prices[current] = {
                "EUR": 1.0,
                **{
                    currency: (0.4 + offset * 0.07)
                    * (1 + 0.0001 * index * (offset + 1))
                    * price_multiplier
                    for offset, currency in enumerate(currencies)
                },
            }
            feature_shift = (
                1000.0 if change_features_after is not None and current > change_features_after else 0.0
            )
            features[current] = {
                **{
                    f"{currency}__log_return_1": index * 0.0001 + offset + feature_shift
                    for offset, currency in enumerate(currencies)
                },
                **{
                    f"{currency}__rsi_14": 0.5 + index * 0.00001 + offset
                    for offset, currency in enumerate(currencies)
                },
            }
            index += 1
        current += timedelta(days=1)
    return MarketData(
        prices_by_date=prices,
        eur_source_file="synthetic-eur.parquet",
        eur_source_sha256=hashlib.sha256(b"synthetic-eur").hexdigest(),
        features_by_date=features,
        feature_names=("log_return_1", "rsi_14"),
        complete_feature_dates=frozenset(features),
        feature_source_file="synthetic-features.parquet",
        feature_source_sha256=hashlib.sha256(b"synthetic-features").hexdigest(),
    )


def configuration(
    *,
    development_fraction: float = 0.75,
    availability_lag_days: int = 7,
    features: tuple[str, ...] = ("log_return_1", "rsi_14"),
    fold_count: int = 3,
    minimum_training_years: int = 5,
    embargo_days: int = 0,
) -> MLDatasetConfig:
    return MLDatasetConfig(
        development_fraction=development_fraction,
        availability_lag_days=availability_lag_days,
        feature_names=features,
        fold_count=fold_count,
        minimum_training_years=minimum_training_years,
        embargo_days=embargo_days,
    )


def test_dataset_uses_friday_cutoff_and_actual_execution_to_execution_targets():
    data = synthetic_market_data()
    result = build_ml_dataset(data, configuration())

    example = next(row for row in result.rows if row["signal_date"] == date(2000, 1, 14))
    assert example["information_cutoff_date"] == date(2000, 1, 7)
    assert example["last_feature_market_date"] == date(2000, 1, 7)
    assert example["execution_date"] == date(2000, 1, 17)
    assert example["next_execution_date"] == date(2000, 1, 24)
    assert (example["signal_date"] - example["last_feature_market_date"]).days >= 7
    expected_usd = (
        data.prices_by_date[example["next_execution_date"]]["USD"]
        / data.prices_by_date[example["execution_date"]]["USD"]
        - 1
    )
    assert example["y_simple_return_USD"] == pytest.approx(expected_usd)
    assert example["y_simple_return_EUR"] == 0.0
    assert len(result.feature_columns) == 14
    assert all("momentum_" not in column for column in result.feature_columns)
    assert not any(column.startswith("price_") for column in result.feature_columns)
    assert not any(column.startswith("y_") for column in result.feature_columns)


def test_missing_friday_observation_executes_on_first_later_market_observation():
    holiday = date(2000, 2, 21)
    data = synthetic_market_data(skip={holiday})
    result = build_ml_dataset(data, configuration())

    example = next(row for row in result.rows if row["signal_date"] == date(2000, 2, 18))
    assert example["execution_date"] == date(2000, 2, 22)
    assert example["next_execution_date"] > example["execution_date"]


def test_split_uses_full_market_calendar_and_never_labels_final_test_period():
    data = synthetic_market_data()
    result = build_ml_dataset(data, configuration())

    assert result.development_end == data.dates[result.split_index - 1]
    assert result.test_start == data.dates[result.split_index]
    assert all(row["next_execution_date"] < result.test_start for row in result.rows)
    assert result.exclusion_counts["no_next_execution_within_development"] >= 1
    assert result.candidate_signal_count == len(result.rows) + sum(result.exclusion_counts.values())


def test_future_mutations_do_not_change_past_features_or_causal_cutoff():
    baseline = synthetic_market_data()
    mutation_start = date(2004, 6, 1)
    changed = synthetic_market_data(
        change_prices_after=mutation_start,
        change_features_after=mutation_start,
    )
    before = build_ml_dataset(baseline, configuration())
    after = build_ml_dataset(changed, configuration())
    baseline_by_signal = {row["signal_date"]: row for row in before.rows}
    changed_by_signal = {row["signal_date"]: row for row in after.rows}
    prior_signals = [
        signal
        for signal in baseline_by_signal
        if baseline_by_signal[signal]["information_cutoff_date"] <= mutation_start
    ]

    assert prior_signals
    for signal in prior_signals:
        assert {
            column: baseline_by_signal[signal][column] for column in before.feature_columns
        } == {
            column: changed_by_signal[signal][column] for column in after.feature_columns
        }
        assert (
            baseline_by_signal[signal]["last_feature_market_date"]
            <= baseline_by_signal[signal]["information_cutoff_date"]
        )


def test_walk_forward_folds_expand_and_purge_overlapping_training_labels():
    first = date(2000, 1, 7)
    rows = []
    for index in range(12 * 52):
        signal = first + timedelta(days=7 * index)
        rows.append(
            {
                "signal_date": signal,
                "next_execution_date": signal + timedelta(days=14),
            }
        )

    folds = build_walk_forward_folds(
        rows,
        fold_count=3,
        minimum_training_years=3,
        embargo_days=2,
    )

    assert len(folds) == 3
    assert len(folds[0].train_indices) < len(folds[1].train_indices) < len(folds[2].train_indices)
    for fold in folds:
        assert fold.purged_training_examples > 0
        assert all(
            rows[index]["next_execution_date"] + timedelta(days=2) < fold.validation_start
            for index in fold.train_indices
        )
        assert max(fold.train_indices) < min(fold.validation_indices)
        assert (fold.validation_start - fold.training_start).days >= 365 * 3


def test_standardizer_is_fit_only_on_training_and_handles_constant_columns():
    training = pd.DataFrame({"varying": [1.0, 3.0, 5.0], "constant": [4.0, 4.0, 4.0]})
    validation = pd.DataFrame({"varying": [1000.0], "constant": [4.0]})

    scaler = FoldStandardizer.fit(training)
    normalized = scaler.transform(validation)

    assert scaler.means == pytest.approx((3.0, 4.0))
    assert scaler.scales == pytest.approx((np.std([1.0, 3.0, 5.0]), 1.0))
    assert scaler.constant_columns == ("constant",)
    assert normalized.loc[0, "varying"] > 100
    assert normalized.loc[0, "constant"] == 0.0
    with pytest.raises(ValueError, match="finite"):
        scaler.transform(pd.DataFrame({"varying": [float("nan")], "constant": [4.0]}))


def test_feature_configuration_rejects_redundant_momentum_columns():
    with pytest.raises(MLDatasetError, match="duplicate"):
        configuration(features=("log_return_5", "momentum_5"))


def test_dataset_artifacts_are_content_addressed_and_keep_x_y_separate(tmp_path):
    data = synthetic_market_data()
    result = build_ml_dataset(data, configuration())
    folds = build_walk_forward_folds(
        result.rows,
        fold_count=3,
        minimum_training_years=5,
    )

    paths = export_ml_dataset(result, tmp_path, folds=folds)
    repeated_paths = export_ml_dataset(result, tmp_path, folds=folds)

    assert paths == repeated_paths
    dataset_path, manifest_path, fold_paths = paths
    table = pq.read_table(dataset_path)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert table.num_rows == len(result.rows)
    assert set(result.feature_columns).issubset(table.column_names)
    assert set(result.target_columns).issubset(table.column_names)
    assert manifest["content_sha256"] == (
        table.schema.metadata[b"forex_v1.ml_dataset.content_sha256"].decode("ascii")
    )
    assert manifest["test_start_excluded"] == result.test_start.isoformat()
    assert manifest["excluded_test_period_used"] is False
    assert manifest["walk_forward"]["fold_count"] == len(folds)
    assert manifest["normalization"]["targets_scaled"] is False
    assert len(fold_paths) == len(folds) * 3
    scaler_doc = json.loads(next(path for path in fold_paths if path.name.endswith("_scaler.json")).read_text())
    train_csv = next(path for path in fold_paths if path.name.endswith("_train.csv"))
    train_normalized = pd.read_csv(train_csv)
    assert scaler_doc["fit_scope"] == "training X rows only"
    assert tuple(scaler_doc["columns"]) == result.feature_columns
    assert all(column in train_normalized for column in result.target_columns)
    assert not train_normalized.loc[:, result.feature_columns].isna().any().any()
