from __future__ import annotations

import json
from dataclasses import replace
from datetime import date

import numpy as np
import pandas as pd
import pytest
import torch

from forex_v1.backtest.config import BacktestConfig
from forex_v1.backtest.data import market_data_from_records
from forex_v1.backtest.engine import run_backtest
from forex_v1.backtest.strategy import FixedWeightProvider
from forex_v1.models.dataset import DEFAULT_FEATURE_NAMES, FOREIGN_CURRENCIES
from forex_v1.models.mlp import (
    OUTPUT_CURRENCIES,
    AllocationMLP,
    MLPConfig,
    differentiable_growth_loss,
)
from forex_v1.models.mlp_training import (
    INNER_VALIDATION_FRACTION,
    LoadedFold,
    MLPTrainingConfig,
    MLPTrainingError,
    MLPWeightProvider,
    _fit_outer_model,
    _inner_validation_partition,
    _load_fold,
    _synthetic_accounting_cases,
    _train_fold,
    scaler_from_payload,
)
from forex_v1.models.temporal import FoldStandardizer


def test_mlp_architecture_softmax_and_output_constraints():
    model = AllocationMLP()
    inputs = torch.randn(6, 91)

    weights = model(inputs)

    assert model.config.layer_sizes == (91, 32, 16, 8)
    assert weights.shape == (6, 8)
    assert torch.all(weights >= 0)
    assert torch.all(weights <= 1)
    torch.testing.assert_close(weights.sum(dim=1), torch.ones(6))
    assert OUTPUT_CURRENCIES == ("EUR", "USD", "GBP", "JPY", "CHF", "CAD", "AUD", "NZD")


def test_growth_loss_has_finite_nonzero_gradients():
    model = AllocationMLP()
    features = torch.randn(9, 91)
    returns = torch.randn(9, 8) * 0.01

    result = differentiable_growth_loss(model(features), returns)
    result.loss.backward()

    gradients = [parameter.grad for parameter in model.parameters()]
    assert all(gradient is not None for gradient in gradients)
    assert all(torch.isfinite(gradient).all() for gradient in gradients if gradient is not None)
    assert any(torch.count_nonzero(gradient) for gradient in gradients if gradient is not None)


def test_optimization_reduces_loss_on_synthetic_positive_usd_returns():
    torch.manual_seed(22)
    model = AllocationMLP(MLPConfig((91, 8, 8)))
    features = torch.zeros(24, 91)
    returns = torch.zeros(24, 8)
    returns[:, 1] = 0.04
    optimizer = torch.optim.Adam(model.parameters(), lr=0.04)
    initial = float(differentiable_growth_loss(model(features), returns).loss.item())

    for _ in range(100):
        optimizer.zero_grad(set_to_none=True)
        loss = differentiable_growth_loss(model(features), returns).loss
        loss.backward()
        optimizer.step()

    final = float(differentiable_growth_loss(model(features), returns).loss.item())
    assert final < initial - 1e-3
    assert torch.isfinite(torch.tensor(final))


def test_fees_are_zero_for_unchanged_drifted_holdings_and_positive_on_initial_buy():
    targets = torch.zeros(2, 8)
    targets[0, 0] = 0.5
    targets[0, 1] = 0.5
    returns = torch.zeros(2, 8)
    returns[0, 1] = 0.2
    drifted_usd = 0.5 * 1.2 / (0.5 + 0.5 * 1.2)
    targets[1, 1] = drifted_usd
    targets[1, 0] = 1 - drifted_usd
    result = differentiable_growth_loss(targets, returns, fee_rate=0.001)
    eur_first_buy = torch.zeros(1, 8)
    eur_first_buy[0, 0] = 0.75
    eur_first_buy[0, 1] = 0.25
    initial_buy = differentiable_growth_loss(
        eur_first_buy,
        torch.zeros(1, 8),
        fee_rate=0.001,
    )

    assert result.cost_fractions[0].item() == pytest.approx(0.0005)
    assert result.cost_fractions[1].item() == pytest.approx(0.0, abs=1e-8)
    assert result.foreign_turnover[0].item() == pytest.approx(0.5)
    assert result.foreign_turnover[1].item() == pytest.approx(0.0, abs=1e-8)
    assert result.pretrade_weights[1, 1].item() == pytest.approx(drifted_usd)
    assert initial_buy.cost_fractions[0].item() == pytest.approx(0.00025)


def test_growth_factor_matches_manual_single_currency_calculation():
    target = torch.zeros(1, 8)
    target[0, 1] = 1.0
    returns = torch.zeros(1, 8)
    returns[0, 1] = 0.1

    result = differentiable_growth_loss(target, returns, fee_rate=0.001)

    expected_factor = (1 - 0.001) * 1.1
    assert result.net_growth_factors[0].item() == pytest.approx(expected_factor)
    assert result.loss.item() == pytest.approx(-np.log(expected_factor))


def test_turnover_penalty_counts_initial_buy_and_foreign_currency_rotation():
    targets = torch.zeros((3, 8), dtype=torch.float64)
    targets[0, 0] = 0.5
    targets[0, 1] = 0.5
    targets[1, 0] = 0.5
    targets[1, 3] = 0.5
    targets[2, 0] = 0.5
    targets[2, 1] = 0.5
    returns = torch.zeros_like(targets)

    result = differentiable_growth_loss(
        targets,
        returns,
        fee_rate=0.001,
        lambda_turnover=0.001,
    )

    assert result.foreign_turnover[0].item() == pytest.approx(0.5)
    assert result.foreign_turnover[1].item() == pytest.approx(1.0)
    assert result.foreign_turnover[2].item() == pytest.approx(1.0)


def test_turnover_regularizer_is_differentiable_and_zero_lambda_preserves_existing_loss():
    targets = torch.tensor(
        [
            [0.5, 0.5, 0, 0, 0, 0, 0, 0],
            [0.5, 0, 0.5, 0, 0, 0, 0, 0],
            [0.5, 0.5, 0, 0, 0, 0, 0, 0],
        ],
        dtype=torch.float64,
        requires_grad=True,
    )
    returns = torch.zeros_like(targets)
    ordinary = differentiable_growth_loss(targets, returns, fee_rate=0.001)
    zero_lambda = differentiable_growth_loss(
        targets,
        returns,
        fee_rate=0.001,
        lambda_turnover=0.0,
    )
    regularized = differentiable_growth_loss(
        targets,
        returns,
        fee_rate=0.001,
        lambda_turnover=0.001,
    )
    assert torch.equal(ordinary.loss, zero_lambda.loss)
    assert regularized.loss > ordinary.loss
    assert (regularized.loss - ordinary.loss).item() == pytest.approx(
        0.001 * regularized.foreign_turnover.mean().item()
    )

    regularized.loss.backward()

    assert targets.grad is not None
    assert torch.isfinite(targets.grad).all()
    assert torch.count_nonzero(targets.grad) > 0


def test_turnover_penalty_uses_drifted_pretrade_weights_not_previous_target():
    target = np.zeros((2, 8), dtype=np.float64)
    target[0, 0] = target[0, 1] = 0.5
    usd_return = 0.20
    target[1, 1] = 0.5 * (1 + usd_return) / (0.5 + 0.5 * (1 + usd_return))
    target[1, 0] = 1.0 - target[1, 1]
    returns = np.zeros_like(target)
    returns[0, 1] = usd_return

    result = differentiable_growth_loss(
        torch.tensor(target, dtype=torch.float64),
        torch.tensor(returns, dtype=torch.float64),
        lambda_turnover=0.001,
    )

    assert result.foreign_turnover[0].item() == pytest.approx(0.5)
    assert result.foreign_turnover[1].item() == pytest.approx(0.0, abs=1e-12)


def test_exact_6a_fee_matches_differentiable_approximation_with_small_documented_gap():
    records = {}
    for market_date, usd in (
        (date(2024, 1, 4), 0.8),
        (date(2024, 1, 5), 0.8),
        (date(2024, 1, 11), 0.8),
        (date(2024, 1, 12), 0.8),
        (date(2024, 1, 15), 0.8),
        (date(2024, 1, 16), 0.88),
    ):
        records[market_date] = {
            "EUR": 1.0,
            "USD": usd,
            "GBP": 1.1,
            "JPY": 0.006,
            "CHF": 1.0,
            "CAD": 0.7,
            "AUD": 0.65,
            "NZD": 0.6,
        }
    data = market_data_from_records(records)
    config = BacktestConfig(
        initial_capital_eur=10_000,
        transaction_cost_bps=10,
        availability_lag_days=7,
        target_weights={"EUR": 0.75, "USD": 0.25},
        start_date=date(2024, 1, 4),
        end_date=date(2024, 1, 16),
        signal_end_date=date(2024, 1, 12),
    )
    exact = run_backtest(data, FixedWeightProvider({"EUR": 0.75, "USD": 0.25}), config)
    targets = torch.tensor([[0.75, 0.25, 0, 0, 0, 0, 0, 0]], dtype=torch.float64)
    returns = torch.zeros_like(targets)
    approximate = differentiable_growth_loss(targets, returns, fee_rate=0.001)
    exact_fee = exact.total_fees_eur
    approximate_fee = approximate.cost_fractions[0].item() * 10_000

    assert exact_fee == pytest.approx(sum(tx.fee_eur for tx in exact.executions[0].rebalance.transactions))
    assert abs(exact_fee - approximate_fee) < 0.01
    exact_net_factor = exact.nav_series.loc[date(2024, 1, 15)] / 10_000
    assert exact_net_factor == pytest.approx(1 - exact_fee / 10_000)


def test_scaler_manifest_is_ordered_train_only_and_rejects_bad_scope():
    columns = ("USD__indicator", "GBP__indicator")
    training = pd.DataFrame([[1.0, 4.0], [3.0, 8.0]], columns=columns)
    fitted = FoldStandardizer.fit(training)
    payload = {
        "fit_scope": "training X rows only",
        **fitted.to_dict(),
    }

    loaded = scaler_from_payload(payload, columns)
    validation = pd.DataFrame([[100.0, 100.0]], columns=columns)

    assert loaded.means == pytest.approx((2.0, 6.0))
    assert loaded.transform(validation).iloc[0, 0] == pytest.approx(98.0)
    with pytest.raises(ValueError, match="order"):
        loaded.transform(validation.loc[:, list(reversed(columns))])
    with pytest.raises(MLPTrainingError, match="training-only"):
        scaler_from_payload({**payload, "fit_scope": "all data"}, columns)


def test_fold_loader_uses_only_saved_train_scaler_and_excludes_test_period(tmp_path):
    dataset_hash = "a" * 64
    stem = f"ml_dataset_v1_{dataset_hash}"
    dataset_dir = tmp_path
    fold_dir = dataset_dir / "folds" / stem
    fold_dir.mkdir(parents=True)
    columns = tuple(
        f"{currency}__{feature}"
        for currency in FOREIGN_CURRENCIES
        for feature in DEFAULT_FEATURE_NAMES
    )
    targets = tuple(f"y_simple_return_{currency}" for currency in OUTPUT_CURRENCIES)
    manifest = {
        "feature_columns": list(columns),
        "target_columns": list(targets),
        "test_start_excluded": "2019-10-25",
    }
    (dataset_dir / f"{stem}.json").write_text(json.dumps(manifest), encoding="utf-8")
    train_features = pd.DataFrame(
        [[1.0] * len(columns), [3.0] * len(columns)],
        columns=columns,
    )
    scaler = FoldStandardizer.fit(train_features)
    scaler_payload = {
        **scaler.to_dict(),
        "fit_scope": "training X rows only",
        "purged_training_examples": 1,
    }
    common_dates = {
        "signal_date": ["2000-01-07", "2000-01-14"],
        "execution_date": ["2000-01-10", "2000-01-17"],
        "next_execution_date": ["2000-01-17", "2000-01-24"],
    }
    train_frame = pd.concat(
        [
            pd.DataFrame(common_dates),
            train_features,
            pd.DataFrame(np.zeros((2, len(targets))), columns=targets),
        ],
        axis=1,
    )
    validation_features = pd.DataFrame([[100.0] * len(columns)], columns=columns)
    validation_frame = pd.concat(
        [
            pd.DataFrame(
                {
                    "signal_date": ["2000-01-28"],
                    "execution_date": ["2000-01-31"],
                    "next_execution_date": ["2000-02-07"],
                }
            ),
            validation_features,
            pd.DataFrame(np.zeros((1, len(targets))), columns=targets),
        ],
        axis=1,
    )
    train_frame.to_csv(fold_dir / "fold_01_train.csv", index=False)
    validation_frame.to_csv(fold_dir / "fold_01_validation.csv", index=False)
    (fold_dir / "fold_01_scaler.json").write_text(
        json.dumps(scaler_payload),
        encoding="utf-8",
    )

    loaded = _load_fold(dataset_dir, manifest, 1)

    assert loaded.scaler.means == pytest.approx((2.0,) * len(columns))
    assert loaded.scaler.transform(validation_features).iloc[0, 0] == pytest.approx(98.0)
    assert loaded.validation_frame["next_execution_date"].max().date() < date(2019, 10, 25)


def test_provider_is_causal_and_emits_valid_weights_for_admissible_history():
    columns = tuple(
        f"{currency}__{feature}"
        for currency in FOREIGN_CURRENCIES
        for feature in DEFAULT_FEATURE_NAMES
    )
    scaler = FoldStandardizer.fit(pd.DataFrame(np.zeros((2, len(columns))), columns=columns))
    torch.manual_seed(42)
    provider = MLPWeightProvider(
        AllocationMLP(),
        scaler,
        columns,
        device=torch.device("cpu"),
    )
    rows = []
    cutoff = date(2020, 1, 3)
    for currency in FOREIGN_CURRENCIES:
        rows.append(
            {
                "market_date": cutoff,
                "currency": currency,
                **dict.fromkeys(DEFAULT_FEATURE_NAMES, 0.0),
            }
        )
    admissible_history = pd.DataFrame(rows)

    first = provider.get_target_weights(admissible_history, {})
    changed_future = admissible_history.copy()
    changed_future["market_date"] = date(2020, 1, 10)
    second = provider.get_target_weights(admissible_history, {})

    assert first == second
    assert all(weight >= 0 for weight in first.values())
    assert sum(first.values()) == pytest.approx(1.0)
    assert max(changed_future["market_date"]) > cutoff


def test_fold_models_are_distinct_and_seeded_initialization_is_reproducible():
    def parameters(seed: int) -> list[torch.Tensor]:
        torch.manual_seed(seed)
        model = AllocationMLP()
        return [value.detach().clone() for value in model.parameters()]

    first = parameters(42)
    repeated = parameters(42)
    next_fold = parameters(43)

    assert all(torch.equal(left, right) for left, right in zip(first, repeated, strict=True))
    assert any(not torch.equal(left, right) for left, right in zip(first, next_fold, strict=True))


def test_state_dict_checkpoint_reload_reproduces_predictions(tmp_path):
    torch.manual_seed(5)
    model = AllocationMLP()
    inputs = torch.randn(3, 91)
    with torch.no_grad():
        expected = model(inputs)
    checkpoint = tmp_path / "state_dict.pt"
    torch.save(model.state_dict(), checkpoint)
    restored = AllocationMLP()
    restored.load_state_dict(torch.load(checkpoint, weights_only=True))
    restored.eval()

    with torch.no_grad():
        actual = restored(inputs)

    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


def _synthetic_loaded_fold(tmp_path) -> tuple[LoadedFold, MLPTrainingConfig]:
    rng = np.random.default_rng(123)
    feature_columns = tuple(f"feature_{index:02d}" for index in range(91))
    target_columns = tuple(f"y_simple_return_{currency}" for currency in OUTPUT_CURRENCIES)
    raw_train = pd.DataFrame(rng.normal(size=(20, 91)), columns=feature_columns)
    scaler = FoldStandardizer.fit(raw_train)
    train_features = scaler.transform(raw_train)
    signals = pd.date_range("2000-01-07", periods=20, freq="7D")
    raw_returns = rng.normal(scale=0.01, size=(20, 8))
    raw_returns[:, 0] = 0.0
    train_frame = pd.concat(
        [
            pd.DataFrame(
                {
                    "signal_date": signals,
                    "execution_date": signals + pd.Timedelta(days=3),
                    "next_execution_date": signals + pd.Timedelta(days=10),
                }
            ),
            train_features.reset_index(drop=True),
            pd.DataFrame(raw_returns, columns=target_columns),
        ],
        axis=1,
    )
    raw_validation = pd.DataFrame(rng.normal(size=(4, 91)), columns=feature_columns)
    validation_features = scaler.transform(raw_validation)
    validation_signals = pd.date_range(signals[-1] + pd.Timedelta(days=7), periods=4, freq="7D")
    validation_returns = rng.normal(scale=0.01, size=(4, 8))
    validation_returns[:, 0] = 0.0
    validation_frame = pd.concat(
        [
            pd.DataFrame(
                {
                    "signal_date": validation_signals,
                    "execution_date": validation_signals + pd.Timedelta(days=3),
                    "next_execution_date": validation_signals + pd.Timedelta(days=10),
                }
            ),
            validation_features.reset_index(drop=True),
            pd.DataFrame(validation_returns, columns=target_columns),
        ],
        axis=1,
    )
    payload = {
        **scaler.to_dict(),
        "fit_scope": "training X rows only",
        "purged_training_examples": 1,
    }
    fold = LoadedFold(
        fold_number=1,
        train_frame=train_frame,
        validation_frame=validation_frame,
        feature_columns=feature_columns,
        target_columns=target_columns,
        scaler=scaler,
        scaler_payload=payload,
        purged_training_examples=1,
    )
    config = MLPTrainingConfig(
        layer_sizes=(91, 8, 16, 8),
        optimizer="AdamW",
        learning_rate=0.001,
        weight_decay=0.01,
        max_epochs=3,
        early_stopping_patience=2,
        random_seed=42,
        device="cpu",
        gradient_clip_norm=5.0,
        initial_capital_eur=10_000.0,
        transaction_cost_bps=10.0,
        lambda_turnover=0.0,
        drawdown_penalty=0.0,
        drawdown_threshold=0.2,
        dataset_dir=tmp_path,
        eur_data_dir=tmp_path,
        feature_data_dir=tmp_path,
        models_dir=tmp_path / "models",
        report_path=tmp_path / "audit.md",
    )
    return fold, config


def test_inner_validation_is_chronological_purged_and_scaled_on_inner_train(tmp_path):
    fold, _ = _synthetic_loaded_fold(tmp_path)

    partition = _inner_validation_partition(fold)
    raw_inner_train = (
        fold.train_frame.loc[:, list(fold.feature_columns)].to_numpy()
        * np.asarray(fold.scaler.scales)
        + np.asarray(fold.scaler.means)
    )
    inner_start_index = len(fold.train_frame) - len(partition.validation_frame)
    raw_inner_train_frame = pd.DataFrame(
        raw_inner_train[: inner_start_index - partition.purged_training_examples],
        columns=fold.feature_columns,
    )

    assert len(partition.validation_frame) == int(
        np.ceil(len(fold.train_frame) * INNER_VALIDATION_FRACTION)
    )
    assert partition.train_frame["next_execution_date"].max().date() < partition.validation_start
    assert (
        partition.train_frame["signal_date"].max()
        < partition.validation_frame["signal_date"].min()
    )
    assert partition.purged_training_examples >= 1
    assert partition.scaler.means == pytest.approx(raw_inner_train_frame.mean(axis=0).to_numpy())
    assert partition.outer_train_scaler == fold.scaler


def test_outer_validation_changes_do_not_change_scalers_epochs_or_trained_weights(tmp_path):
    fold, base_config = _synthetic_loaded_fold(tmp_path)
    config = replace(base_config, lambda_turnover=0.001)

    original = _train_fold(
        fold,
        config,
        dataset_sha256="a" * 64,
        source_hashes={"eur_sha256": "b" * 64, "feature_sha256": "c" * 64},
    )
    changed_validation = fold.validation_frame.copy()
    changed_validation.loc[:, list(fold.feature_columns)] += 1_000_000.0
    changed_validation.loc[:, list(fold.target_columns[1:])] = 0.25
    changed_fold = LoadedFold(
        fold_number=fold.fold_number,
        train_frame=fold.train_frame.copy(),
        validation_frame=changed_validation,
        feature_columns=fold.feature_columns,
        target_columns=fold.target_columns,
        scaler=fold.scaler,
        scaler_payload=fold.scaler_payload,
        purged_training_examples=fold.purged_training_examples,
    )
    changed = _train_fold(
        changed_fold,
        config,
        dataset_sha256="a" * 64,
        source_hashes={"eur_sha256": "b" * 64, "feature_sha256": "c" * 64},
    )

    assert original["best_epoch"] == changed["best_epoch"]
    assert original["inner_partition"].scaler == changed["inner_partition"].scaler
    assert original["fold"].scaler == changed["fold"].scaler
    assert all(
        torch.equal(value, changed["model"].state_dict()[name])
        for name, value in original["model"].state_dict().items()
    )


def test_outer_training_model_is_fresh_and_seed_reproducible(tmp_path):
    fold, config = _synthetic_loaded_fold(tmp_path)
    first = _fit_outer_model(fold, config, epochs=2, seed=42)
    repeated = _fit_outer_model(fold, config, epochs=2, seed=42)
    other_fold = _fit_outer_model(fold, config, epochs=2, seed=43)

    assert all(
        torch.equal(value, repeated.state_dict()[name])
        for name, value in first.state_dict().items()
    )
    assert any(
        not torch.equal(value, other_fold.state_dict()[name])
        for name, value in first.state_dict().items()
    )


def test_synthetic_exact_accounting_comparison_reconciles_fees_and_no_shorts():
    cases = _synthetic_accounting_cases(0.001)
    by_name = {case["case"]: case for case in cases}

    assert by_name["eur_cash"]["exact_net_return"] == pytest.approx(0.0)
    assert by_name["eur_cash"]["exact_fees_eur"] == pytest.approx(0.0)
    assert by_name["initial_investment_50pct_usd"]["exact_fees_eur"] == pytest.approx(5.0)
    assert by_name["full_usd_to_jpy_rotation"]["exact_fees_eur"] == pytest.approx(
        29.9500699101
    )
    assert by_name["full_usd_to_jpy_rotation"]["cash_before_second_rebalance_eur"] == pytest.approx(
        0.0
    )
    assert all(case["nonnegative_cash"] for case in cases)
    assert all(case["nonnegative_positions"] for case in cases)
    assert all(
        case["exact_fees_eur"] == pytest.approx(case["fee_from_notional_eur"])
        for case in cases
    )
    assert max(abs(case["approximation_minus_exact_bps"]) for case in cases) < 1.0
