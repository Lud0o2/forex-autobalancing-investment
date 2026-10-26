from __future__ import annotations

import hashlib
from datetime import date, timedelta

import numpy as np
import pandas as pd
import pytest

from forex_v1.models.signal_diagnostic import (
    MLSignalDiagnosticError,
    _correlation,
    _mean_prediction,
    _spearman,
    _split_and_scale_inner_train,
    fit_ridge,
    predict_ridge,
    sha256_file,
    validate_8a_alignment,
    validate_finite_features,
)
from forex_v1.models.temporal import FoldStandardizer


def _synthetic_8a_rows() -> pd.DataFrame:
    signal = date(2020, 1, 10)
    row: dict[str, object] = {
        "signal_date": signal,
        "information_cutoff_date": signal - timedelta(days=7),
        "last_feature_market_date": signal - timedelta(days=7),
        "execution_date": signal + timedelta(days=3),
        "next_execution_date": signal + timedelta(days=10),
        "X__indicator": 0.5,
    }
    for currency, target in zip(
        ("USD", "GBP", "JPY", "CHF", "CAD", "AUD", "NZD"),
        (0.1, -0.1, 0.02, 0.03, -0.02, 0.04, -0.05),
        strict=True,
    ):
        row[f"price_start_eur_per_unit_{currency}"] = 2.0
        row[f"price_end_eur_per_unit_{currency}"] = 2.0 * (1.0 + target)
        row[f"y_simple_return_{currency}"] = target
    row["y_simple_return_EUR"] = 0.0
    return pd.DataFrame([row])


def test_future_returns_match_the_8a_execution_to_next_execution_prices():
    rows = _synthetic_8a_rows()

    validate_8a_alignment(
        rows,
        feature_columns=("X__indicator",),
        target_columns=(
            "y_simple_return_EUR",
            *(f"y_simple_return_{currency}" for currency in ("USD", "GBP", "JPY", "CHF", "CAD", "AUD", "NZD")),
        ),
        development_end=date(2020, 12, 31),
        test_start=date(2021, 1, 1),
        availability_lag_days=7,
    )

    rows.loc[0, "price_end_eur_per_unit_USD"] *= 1.1
    with pytest.raises(MLSignalDiagnosticError, match="does not match execution prices for USD"):
        validate_8a_alignment(
            rows,
            feature_columns=("X__indicator",),
            target_columns=(
                "y_simple_return_EUR",
                *(f"y_simple_return_{currency}" for currency in ("USD", "GBP", "JPY", "CHF", "CAD", "AUD", "NZD")),
            ),
            development_end=date(2020, 12, 31),
            test_start=date(2021, 1, 1),
            availability_lag_days=7,
        )


def test_correlation_detects_a_strong_synthetic_signal():
    rng = np.random.default_rng(21)
    feature = rng.normal(size=400)
    target = 0.8 * feature + rng.normal(scale=0.1, size=len(feature))

    assert _correlation(feature, target) > 0.98
    assert _spearman(feature, target) > 0.98


def test_correlation_is_near_zero_for_synthetic_noise():
    rng = np.random.default_rng(7)
    feature = rng.normal(size=8000)
    unrelated_target = rng.normal(size=8000)

    assert abs(_correlation(feature, unrelated_target)) < 0.04
    assert abs(_spearman(feature, unrelated_target)) < 0.04


def test_correlations_are_undefined_for_constant_inputs():
    assert _correlation(np.ones(10), np.arange(10, dtype=np.float64)) is None
    assert _spearman(np.ones(10), np.arange(10, dtype=np.float64)) is None


def test_ridge_supports_multiple_outputs_and_reproduces_predictions():
    rng = np.random.default_rng(42)
    features = rng.normal(size=(120, 6))
    targets = np.column_stack(
        (
            0.7 * features[:, 0] - 0.2 * features[:, 3],
            -0.5 * features[:, 1] + 0.4 * features[:, 4],
            0.3 * features[:, 2] + 0.1 * features[:, 5],
        )
    )

    coefficients, intercept = fit_ridge(features, targets, alpha=10.0)
    first = predict_ridge(features[:10], coefficients, intercept)
    second_coefficients, second_intercept = fit_ridge(features, targets, alpha=10.0)
    second = predict_ridge(features[:10], second_coefficients, second_intercept)

    assert coefficients.shape == (6, 3)
    assert intercept.shape == (3,)
    assert first.shape == (10, 3)
    assert np.array_equal(first, second)


def test_ridge_improves_on_training_mean_when_a_real_signal_exists():
    rng = np.random.default_rng(13)
    train_x = rng.normal(size=(500, 5))
    validation_x = rng.normal(size=(250, 5))
    weights = np.array([[0.8, -0.4], [0.0, 0.0], [0.2, 0.0], [0.0, 0.6], [0.0, 0.0]])
    train_y = train_x @ weights + rng.normal(scale=0.2, size=(500, 2))
    validation_y = validation_x @ weights + rng.normal(scale=0.2, size=(250, 2))
    coefficients, intercept = fit_ridge(train_x, train_y, alpha=10.0)

    ridge_mse = np.square(predict_ridge(validation_x, coefficients, intercept) - validation_y).mean()
    constant_mse = np.square(_mean_prediction(train_y)[: len(validation_y)] - validation_y).mean()

    assert ridge_mse < constant_mse * 0.2


def test_constant_features_are_reported_and_standardize_to_zero():
    training = pd.DataFrame({"constant": [3.0, 3.0, 3.0], "varying": [1.0, 2.0, 3.0]})
    scaler = FoldStandardizer.fit(training)

    transformed = scaler.transform(training)

    assert scaler.constant_columns == ("constant",)
    assert np.array_equal(transformed["constant"].to_numpy(), np.zeros(3))


@pytest.mark.parametrize("bad_value", [np.nan, np.inf, -np.inf])
def test_missing_or_nonfinite_features_are_rejected_without_imputation(bad_value: float):
    features = pd.DataFrame({"feature": [0.0, bad_value]})

    with pytest.raises(MLSignalDiagnosticError, match="missing or non-finite"):
        validate_finite_features(features)


def test_scaler_is_fitted_only_on_training_features():
    train = pd.DataFrame({"feature": [1.0, 3.0, 5.0]})
    future_validation = pd.DataFrame({"feature": [1000.0]})
    scaler = FoldStandardizer.fit(train)
    validation_scaled = scaler.transform(future_validation)

    assert scaler.means == (3.0,)
    assert scaler.scales == (np.sqrt(8.0 / 3.0),)
    assert validation_scaled.iloc[0, 0] > 600


def test_inner_split_purges_labels_that_reach_the_validation_boundary():
    signals = [date(2020, 1, 3) + timedelta(days=7 * index) for index in range(10)]
    raw_features = pd.DataFrame({"feature": np.arange(10, dtype=np.float64)})
    outer_scaler = FoldStandardizer.fit(raw_features)
    frame = pd.DataFrame(
        {
            "signal_date": pd.to_datetime(signals),
            "execution_date": pd.to_datetime([value + timedelta(days=3) for value in signals]),
            "next_execution_date": pd.to_datetime(
                [value + timedelta(days=10) for value in signals]
            ),
            "feature": outer_scaler.transform(raw_features)["feature"].to_numpy(),
            "target": np.arange(10, dtype=np.float64),
        }
    )

    train, validation, _, _, scaler, boundary, purged = _split_and_scale_inner_train(
        frame,
        ("feature",),
        outer_scaler,
        fold_number=1,
    )

    assert validation["signal_date"].min().date() == boundary
    assert train["next_execution_date"].max().date() < boundary
    assert purged == 1
    assert scaler.means == (3.0,)


def test_future_inputs_are_rejected_even_when_feature_values_are_finite():
    rows = _synthetic_8a_rows()
    rows.loc[0, "last_feature_market_date"] = rows.loc[0, "signal_date"]

    with pytest.raises(MLSignalDiagnosticError, match="after its information cutoff"):
        validate_8a_alignment(
            rows,
            feature_columns=("X__indicator",),
            target_columns=(
                "y_simple_return_EUR",
                *(f"y_simple_return_{currency}" for currency in ("USD", "GBP", "JPY", "CHF", "CAD", "AUD", "NZD")),
            ),
            development_end=date(2020, 12, 31),
            test_start=date(2021, 1, 1),
            availability_lag_days=7,
        )


def test_hashing_diagnostic_inputs_is_read_only(tmp_path):
    input_path = tmp_path / "dataset.parquet"
    original = b"immutable 8A test bytes"
    input_path.write_bytes(original)
    before = hashlib.sha256(original).hexdigest()

    assert sha256_file(input_path) == before
    assert input_path.read_bytes() == original
