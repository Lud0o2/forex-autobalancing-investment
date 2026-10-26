from __future__ import annotations

import json
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
    MLPTrainingError,
    MLPWeightProvider,
    _load_fold,
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
