"""Train and validate fold-specific allocation MLPs on development data only."""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import yaml
from torch import Tensor

from forex_v1.backtest.config import BacktestConfig
from forex_v1.backtest.data import MarketData, load_market_data
from forex_v1.backtest.engine import BacktestResult, run_backtest
from forex_v1.data.layout import PROJECT_ROOT
from forex_v1.data.storage._files import write_immutable_file
from forex_v1.evaluation.metrics import calculate_metrics
from forex_v1.models.mlp import (
    OUTPUT_CURRENCIES,
    AllocationMLP,
    GrowthLossResult,
    MLPConfig,
    differentiable_growth_loss,
)
from forex_v1.models.temporal import FoldStandardizer
from forex_v1.strategies import create_benchmark_provider

BENCHMARKS = ("eur_cash", "equal_weight_weekly", "buy_hold_equal_weight")
FOLD_NAMES = ("fold_01", "fold_02", "fold_03")
DEFAULT_DATASET_MANIFEST_VERSION = 1


class MLPTrainingError(RuntimeError):
    """Raised when fold training or exact validation cannot be completed safely."""


@dataclass(frozen=True, slots=True)
class MLPTrainingConfig:
    """Validated configuration for the first allocation MLP experiment."""

    layer_sizes: tuple[int, ...]
    optimizer: str
    learning_rate: float
    weight_decay: float
    max_epochs: int
    early_stopping_patience: int
    random_seed: int
    device: str
    gradient_clip_norm: float
    initial_capital_eur: float
    transaction_cost_bps: float
    drawdown_penalty: float
    drawdown_threshold: float
    dataset_dir: Path
    eur_data_dir: Path
    feature_data_dir: Path
    models_dir: Path
    report_path: Path

    @property
    def model(self) -> MLPConfig:
        return MLPConfig(self.layer_sizes)

    @property
    def fee_rate(self) -> float:
        return self.transaction_cost_bps / 10_000.0

    def __post_init__(self) -> None:
        if not self.layer_sizes:
            raise ValueError("layer_sizes must define an input, hidden and output layer")
        MLPConfig(self.layer_sizes)
        if self.optimizer != "AdamW":
            raise ValueError("Only the configured AdamW optimizer is supported in this version")
        for field_name, value in (
            ("learning_rate", self.learning_rate),
            ("gradient_clip_norm", self.gradient_clip_norm),
        ):
            if not math.isfinite(value) or value <= 0:
                raise ValueError(f"{field_name} must be finite and positive")
        if not math.isfinite(self.weight_decay) or self.weight_decay < 0:
            raise ValueError("weight_decay must be finite and non-negative")
        for field_name, value, minimum in (
            ("max_epochs", self.max_epochs, 1),
            ("early_stopping_patience", self.early_stopping_patience, 1),
            ("random_seed", self.random_seed, 0),
        ):
            if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
                raise ValueError(f"{field_name} must be an integer >= {minimum}")
        if self.device not in {"cpu", "cuda"}:
            raise ValueError("device must be 'cpu' or 'cuda'")
        if self.device == "cuda" and not torch.cuda.is_available():
            raise ValueError("CUDA was requested but is not available")
        if not math.isfinite(self.initial_capital_eur) or self.initial_capital_eur <= 0:
            raise ValueError("initial_capital_eur must be finite and positive")
        if not math.isfinite(self.transaction_cost_bps) or not 0 <= self.transaction_cost_bps < 10_000:
            raise ValueError("transaction_cost_bps must be finite and in [0, 10000)")
        if not math.isfinite(self.drawdown_penalty) or self.drawdown_penalty < 0:
            raise ValueError("drawdown_penalty must be finite and non-negative")
        if not math.isfinite(self.drawdown_threshold) or not 0 < self.drawdown_threshold < 1:
            raise ValueError("drawdown_threshold must be finite and in (0, 1)")


@dataclass(frozen=True, slots=True)
class LoadedFold:
    """A prepared chronological 8A fold and its train-only scaler."""

    fold_number: int
    train_frame: pd.DataFrame
    validation_frame: pd.DataFrame
    feature_columns: tuple[str, ...]
    target_columns: tuple[str, ...]
    scaler: FoldStandardizer
    scaler_payload: Mapping[str, Any]
    purged_training_examples: int


class MLPWeightProvider:
    """Adapt one trained model and fold scaler to TargetWeightProvider."""

    def __init__(
        self,
        model: AllocationMLP,
        scaler: FoldStandardizer,
        feature_columns: Sequence[str],
        *,
        device: torch.device,
    ) -> None:
        self.model = model.to(device)
        self.model.eval()
        self.scaler = scaler
        self.feature_columns = tuple(feature_columns)
        self.device = device

    def get_target_weights(
        self,
        history_until_t: pd.DataFrame,
        current_state: Mapping[str, object],
    ) -> Mapping[str, float]:
        del current_state
        if history_until_t.empty:
            raise MLPTrainingError("The MLP provider received an empty admissible history")
        if not {"market_date", "currency"}.issubset(history_until_t.columns):
            raise MLPTrainingError("The MLP provider history lacks market_date or currency")
        by_currency: dict[str, set[str]] = {currency: set() for currency in OUTPUT_CURRENCIES[1:]}
        for column in self.feature_columns:
            currency, feature_name = column.split("__", maxsplit=1)
            by_currency[currency].add(feature_name)
        candidates = sorted(history_until_t["market_date"].unique(), reverse=True)
        raw_values: dict[str, float] | None = None
        for market_date in candidates:
            snapshot: dict[str, float] = {}
            for currency, names in by_currency.items():
                matching = history_until_t.loc[
                    (history_until_t["market_date"] == market_date)
                    & (history_until_t["currency"] == currency)
                ]
                if len(matching) != 1 or not names.issubset(matching.columns):
                    snapshot = {}
                    break
                for column in self.feature_columns:
                    column_currency, feature_name = column.split("__", maxsplit=1)
                    if column_currency != currency:
                        continue
                    value = matching.iloc[0][feature_name]
                    if pd.isna(value) or not math.isfinite(float(value)):
                        snapshot = {}
                        break
                    snapshot[column] = float(value)
                if not snapshot and self.feature_columns:
                    break
            if len(snapshot) == len(self.feature_columns):
                raw_values = snapshot
                break
        if raw_values is None:
            raise MLPTrainingError("No complete causal feature snapshot is present in provider history")
        raw = pd.DataFrame([[raw_values[column] for column in self.feature_columns]], columns=self.feature_columns)
        normalized = self.scaler.transform(raw)
        inputs = torch.as_tensor(
            normalized.to_numpy(copy=True),
            dtype=torch.float32,
            device=self.device,
        )
        with torch.no_grad():
            raw_weights = [float(value) for value in self.model(inputs)[0].cpu().tolist()]
        if len(raw_weights) != len(OUTPUT_CURRENCIES):
            raise MLPTrainingError("The MLP output size does not match the eight-currency contract")
        total = math.fsum(raw_weights)
        if not math.isfinite(total) or total <= 0:
            raise MLPTrainingError("The MLP emitted invalid allocation weights")
        weights = [value / total for value in raw_weights]
        weights[0] += 1.0 - math.fsum(weights)
        if weights[0] < 0:
            raise MLPTrainingError("Numerical normalization produced a negative EUR weight")
        return dict(zip(OUTPUT_CURRENCIES, weights, strict=True))


def load_mlp_training_config(path: str | Path) -> MLPTrainingConfig:
    """Load and validate the model/training/finance/data YAML sections."""
    try:
        document = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError) as error:
        raise ValueError(f"Cannot read MLP configuration {path}: {error}") from error
    if not isinstance(document, Mapping):
        raise TypeError("MLP configuration root must be a mapping")
    model = document.get("model")
    training = document.get("training")
    finance = document.get("finance")
    data = document.get("data")
    if not all(isinstance(section, Mapping) for section in (model, training, finance, data)):
        raise ValueError("MLP config must contain model, training, finance and data mappings")
    base = Path(path).resolve().parent.parent

    def resolved(field: str) -> Path:
        value = data.get(field)
        if not isinstance(value, str) or not value:
            raise ValueError(f"data.{field} must be a non-empty relative or absolute path")
        candidate = Path(value)
        return candidate if candidate.is_absolute() else base / candidate

    layer_sizes = model.get("layer_sizes")
    if not isinstance(layer_sizes, list) or any(not isinstance(value, int) for value in layer_sizes):
        raise ValueError("model.layer_sizes must be a list of integers")
    config = MLPTrainingConfig(
        layer_sizes=tuple(layer_sizes),
        optimizer=training.get("optimizer"),
        learning_rate=training.get("learning_rate"),
        weight_decay=training.get("weight_decay"),
        max_epochs=training.get("max_epochs"),
        early_stopping_patience=training.get("early_stopping_patience"),
        random_seed=training.get("random_seed"),
        device=training.get("device"),
        gradient_clip_norm=training.get("gradient_clip_norm"),
        initial_capital_eur=finance.get("initial_capital_eur"),
        transaction_cost_bps=finance.get("transaction_cost_bps"),
        drawdown_penalty=finance.get("drawdown_penalty"),
        drawdown_threshold=finance.get("drawdown_threshold"),
        dataset_dir=resolved("dataset_dir"),
        eur_data_dir=resolved("eur_data_dir"),
        feature_data_dir=resolved("feature_data_dir"),
        models_dir=resolved("models_dir"),
        report_path=resolved("report_path"),
    )
    return config


def _unique_dataset_artifacts(dataset_dir: Path) -> tuple[Path, Path, dict[str, Any]]:
    manifests = sorted(dataset_dir.glob("ml_dataset_v1_*.json"))
    if len(manifests) != 1:
        raise MLPTrainingError(
            f"Expected exactly one validated 8A manifest in {dataset_dir}; found {len(manifests)}"
        )
    manifest_path = manifests[0]
    dataset_path = manifest_path.with_suffix(".parquet")
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise MLPTrainingError(f"Cannot read 8A manifest {manifest_path}: {error}") from error
    if manifest.get("excluded_test_period_used") is not False:
        raise MLPTrainingError("The 8A dataset manifest does not certify test-period exclusion")
    if len(manifest.get("feature_columns", [])) != 91 or len(manifest.get("target_columns", [])) != 8:
        raise MLPTrainingError("The 8A manifest must declare 91 features and eight targets")
    return dataset_path, manifest_path, manifest


def _load_fold(dataset_dir: Path, manifest: Mapping[str, Any], fold_number: int) -> LoadedFold:
    manifest_files = sorted(dataset_dir.glob("ml_dataset_v1_*.json"))
    fold_root = dataset_dir / "folds" / manifest_files[0].stem
    prefix = f"fold_{fold_number:02d}"
    train_path = fold_root / f"{prefix}_train.csv"
    validation_path = fold_root / f"{prefix}_validation.csv"
    scaler_path = fold_root / f"{prefix}_scaler.json"
    try:
        train = pd.read_csv(train_path, parse_dates=["signal_date", "execution_date", "next_execution_date"])
        validation = pd.read_csv(
            validation_path,
            parse_dates=["signal_date", "execution_date", "next_execution_date"],
        )
        scaler_payload = json.loads(scaler_path.read_text(encoding="utf-8"))
    except (OSError, ValueError, json.JSONDecodeError) as error:
        raise MLPTrainingError(f"Cannot read 8A fold {fold_number}: {error}") from error
    feature_columns = tuple(manifest["feature_columns"])
    target_columns = tuple(manifest["target_columns"])
    expected_order = ("signal_date", "execution_date", "next_execution_date", *feature_columns, *target_columns)
    for label, frame in (("train", train), ("validation", validation)):
        if tuple(frame.columns) != expected_order:
            raise MLPTrainingError(f"Fold {fold_number} {label} columns differ from the 8A schema")
        if frame.empty or not np.isfinite(frame.loc[:, (*feature_columns, *target_columns)].to_numpy()).all():
            raise MLPTrainingError(f"Fold {fold_number} {label} data is empty or non-finite")
    if not train["signal_date"].is_monotonic_increasing or not validation["signal_date"].is_monotonic_increasing:
        raise MLPTrainingError(f"Fold {fold_number} examples are not chronological")
    if train["next_execution_date"].max() >= validation["signal_date"].min():
        raise MLPTrainingError(f"Fold {fold_number} contains an unpurged training label")
    if validation["next_execution_date"].max().date().isoformat() >= manifest["test_start_excluded"]:
        raise MLPTrainingError(f"Fold {fold_number} uses a label outside development")
    if scaler_payload.get("fit_scope") != "training X rows only":
        raise MLPTrainingError(f"Fold {fold_number} scaler was not fitted on training X only")
    if tuple(scaler_payload.get("columns", ())) != feature_columns:
        raise MLPTrainingError(f"Fold {fold_number} scaler feature order differs from the dataset")
    scaler = scaler_from_payload(scaler_payload, feature_columns)
    return LoadedFold(
        fold_number=fold_number,
        train_frame=train,
        validation_frame=validation,
        feature_columns=feature_columns,
        target_columns=target_columns,
        scaler=scaler,
        scaler_payload=scaler_payload,
        purged_training_examples=int(scaler_payload["purged_training_examples"]),
    )


def scaler_from_payload(
    payload: Mapping[str, Any],
    feature_columns: Sequence[str],
) -> FoldStandardizer:
    """Reconstruct a scaler only when its declared ordered columns match the fold."""
    columns = tuple(feature_columns)
    if payload.get("fit_scope") != "training X rows only":
        raise MLPTrainingError("Scaler manifest must declare training-only fit scope")
    if tuple(payload.get("columns", ())) != columns:
        raise MLPTrainingError("Scaler columns do not match the fold feature order")
    means = payload.get("means")
    scales = payload.get("scales")
    if not isinstance(means, Mapping) or not isinstance(scales, Mapping):
        raise MLPTrainingError("Scaler manifest must contain means and scales by feature")
    try:
        result = FoldStandardizer(
            columns=columns,
            means=tuple(float(means[column]) for column in columns),
            scales=tuple(float(scales[column]) for column in columns),
            constant_columns=tuple(payload.get("constant_columns", ())),
        )
    except (KeyError, TypeError, ValueError) as error:
        raise MLPTrainingError(f"Scaler manifest has invalid parameters: {error}") from error
    if (
        any(not math.isfinite(value) for value in result.means)
        or any(not math.isfinite(value) or value <= 0 for value in result.scales)
    ):
        raise MLPTrainingError("Scaler parameters must be finite with strictly positive scales")
    return result


def _tensor(frame: pd.DataFrame, columns: Sequence[str], device: torch.device) -> Tensor:
    values = frame.loc[:, list(columns)].to_numpy(dtype=np.float32, copy=True)
    if not np.isfinite(values).all():
        raise MLPTrainingError("Fold inputs contain non-finite values")
    return torch.as_tensor(values, dtype=torch.float32, device=device)


def _loss(
    model: AllocationMLP,
    features: Tensor,
    returns: Tensor,
    config: MLPTrainingConfig,
) -> GrowthLossResult:
    predictions = model(features)
    return differentiable_growth_loss(
        predictions,
        returns,
        fee_rate=config.fee_rate,
        drawdown_penalty=config.drawdown_penalty,
        drawdown_threshold=config.drawdown_threshold,
    )


def _seed_everything(seed: int) -> None:
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True)
    torch.set_num_threads(1)


def _train_fold(
    fold: LoadedFold,
    config: MLPTrainingConfig,
    *,
    dataset_sha256: str,
    source_hashes: Mapping[str, str],
) -> dict[str, Any]:
    device = torch.device(config.device)
    seed = config.random_seed + fold.fold_number - 1
    _seed_everything(seed)
    model = AllocationMLP(config.model).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )
    x_train = _tensor(fold.train_frame, fold.feature_columns, device)
    y_train = _tensor(fold.train_frame, fold.target_columns, device)
    x_validation = _tensor(fold.validation_frame, fold.feature_columns, device)
    y_validation = _tensor(fold.validation_frame, fold.target_columns, device)
    history: list[dict[str, float]] = []
    best_validation = math.inf
    best_epoch = 0
    best_state: dict[str, Tensor] | None = None
    epochs_without_improvement = 0

    for epoch in range(1, config.max_epochs + 1):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        train_result = _loss(model, x_train, y_train, config)
        if not bool(torch.isfinite(train_result.loss)):
            raise MLPTrainingError(f"Fold {fold.fold_number} produced a non-finite training loss")
        train_result.loss.backward()
        gradients = [
            parameter.grad
            for parameter in model.parameters()
            if parameter.grad is not None
        ]
        if not gradients or not all(bool(torch.isfinite(gradient).all()) for gradient in gradients):
            raise MLPTrainingError(f"Fold {fold.fold_number} gradients are missing or non-finite")
        if not any(bool((gradient != 0).any()) for gradient in gradients):
            raise MLPTrainingError(f"Fold {fold.fold_number} gradients are all zero")
        torch.nn.utils.clip_grad_norm_(model.parameters(), config.gradient_clip_norm)
        optimizer.step()

        model.eval()
        with torch.no_grad():
            evaluated_train = _loss(model, x_train, y_train, config)
            validation_result = _loss(model, x_validation, y_validation, config)
        train_value = float(evaluated_train.loss.item())
        validation_value = float(validation_result.loss.item())
        if not math.isfinite(train_value) or not math.isfinite(validation_value):
            raise MLPTrainingError(f"Fold {fold.fold_number} produced non-finite epoch metrics")
        history.append(
            {
                "epoch": epoch,
                "train_loss": train_value,
                "validation_loss": validation_value,
            }
        )
        if validation_value < best_validation:
            best_validation = validation_value
            best_epoch = epoch
            best_state = {
                key: value.detach().cpu().clone()
                for key, value in model.state_dict().items()
            }
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
        if epochs_without_improvement >= config.early_stopping_patience:
            break
    if best_state is None:
        raise MLPTrainingError(f"Fold {fold.fold_number} never produced a valid checkpoint")
    model.load_state_dict(best_state)
    model.eval()

    checkpoint_buffer = io.BytesIO()
    torch.save(best_state, checkpoint_buffer)
    checkpoint_bytes = checkpoint_buffer.getvalue()
    checkpoint_hash = hashlib.sha256(checkpoint_bytes).hexdigest()
    fold_directory = config.models_dir / f"fold_{fold.fold_number:02d}"
    checkpoint_path = fold_directory / f"state_dict_{checkpoint_hash}.pt"
    write_immutable_file(checkpoint_path, checkpoint_bytes)

    checkpoint_metadata = {
        "schema_version": 1,
        "fold_number": fold.fold_number,
        "checkpoint_sha256": checkpoint_hash,
        "dataset_sha256": dataset_sha256,
        "source_hashes_sha256": dict(source_hashes),
        "seed": seed,
        "model": {
            "layer_sizes": list(config.layer_sizes),
            "hidden_activation": "ReLU",
            "output_activation": "Softmax",
            "output_currencies": list(OUTPUT_CURRENCIES),
            "feature_columns": list(fold.feature_columns),
        },
        "training": {
            "optimizer": config.optimizer,
            "learning_rate": config.learning_rate,
            "weight_decay": config.weight_decay,
            "max_epochs": config.max_epochs,
            "early_stopping_patience": config.early_stopping_patience,
            "best_epoch": best_epoch,
            "epochs_run": len(history),
            "best_validation_loss": best_validation,
            "history": history,
        },
        "finance": {
            "initial_capital_eur": config.initial_capital_eur,
            "transaction_cost_bps": config.transaction_cost_bps,
            "drawdown_penalty": config.drawdown_penalty,
            "drawdown_threshold": config.drawdown_threshold,
            "loss": "negative mean log approximate net growth",
        },
        "scaler": fold.scaler_payload,
        "purged_training_examples": fold.purged_training_examples,
        "train_dates": {
            "first_signal": fold.train_frame["signal_date"].min().date().isoformat(),
            "last_signal": fold.train_frame["signal_date"].max().date().isoformat(),
            "last_label_end": fold.train_frame["next_execution_date"].max().date().isoformat(),
        },
        "validation_dates": {
            "first_signal": fold.validation_frame["signal_date"].min().date().isoformat(),
            "last_signal": fold.validation_frame["signal_date"].max().date().isoformat(),
            "last_label_end": fold.validation_frame["next_execution_date"].max().date().isoformat(),
        },
    }
    manifest_path = fold_directory / f"model_{checkpoint_hash}.json"
    write_immutable_file(
        manifest_path,
        (json.dumps(checkpoint_metadata, indent=2, sort_keys=True, allow_nan=False) + "\n").encode(
            "utf-8"
        ),
    )

    reloaded_state = torch.load(io.BytesIO(checkpoint_bytes), map_location=device, weights_only=True)
    reloaded_model = AllocationMLP(config.model).to(device)
    reloaded_model.load_state_dict(reloaded_state)
    reloaded_model.eval()
    with torch.no_grad():
        predictions = reloaded_model(x_validation)
        original_model = AllocationMLP(config.model).to(device)
        original_model.load_state_dict(best_state)
        original_model.eval()
        original_predictions = original_model(x_validation)
    if not torch.equal(predictions, original_predictions):
        raise MLPTrainingError(f"Fold {fold.fold_number} checkpoint predictions did not reproduce")

    with torch.no_grad():
        train_weights = reloaded_model(x_train).cpu().numpy()
        validation_weights = predictions.cpu().numpy()
    train_returns = y_train.cpu().numpy()
    validation_returns = y_validation.cpu().numpy()
    train_approx = differentiable_growth_loss(
        torch.as_tensor(train_weights),
        torch.as_tensor(train_returns),
        fee_rate=config.fee_rate,
        drawdown_penalty=config.drawdown_penalty,
        drawdown_threshold=config.drawdown_threshold,
    )
    validation_approx = differentiable_growth_loss(
        torch.as_tensor(validation_weights),
        torch.as_tensor(validation_returns),
        fee_rate=config.fee_rate,
        drawdown_penalty=config.drawdown_penalty,
        drawdown_threshold=config.drawdown_threshold,
    )
    return {
        "fold": fold,
        "model": reloaded_model,
        "checkpoint_path": checkpoint_path,
        "manifest_path": manifest_path,
        "history": history,
        "best_epoch": best_epoch,
        "best_validation_loss": best_validation,
        "train_weights": train_weights,
        "validation_weights": validation_weights,
        "train_loss_best_model": float(train_approx.loss.item()),
        "validation_loss_best_model": float(validation_approx.loss.item()),
        "training_loss_decreased": bool(
            history[-1]["train_loss"] < history[0]["train_loss"]
        ),
        "validation_loss_worsened_after_best": bool(
            history[-1]["validation_loss"] > best_validation
        ),
    }


def _slice_market_data(market_data: MarketData, end_date: date) -> MarketData:
    dates = tuple(market_date for market_date in market_data.dates if market_date <= end_date)
    return MarketData(
        prices_by_date={market_date: market_data.prices_by_date[market_date] for market_date in dates},
        eur_source_file=market_data.eur_source_file,
        eur_source_sha256=market_data.eur_source_sha256,
        features_by_date={
            market_date: market_data.features_by_date[market_date]
            for market_date in dates
            if market_date in market_data.features_by_date
        },
        feature_names=market_data.feature_names,
        complete_feature_dates=frozenset(
            market_date for market_date in market_data.complete_feature_dates if market_date <= end_date
        ),
        feature_source_file=market_data.feature_source_file,
        feature_source_sha256=market_data.feature_source_sha256,
    )


def _transaction_frame(result: BacktestResult) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "execution_date": execution.execution_date.isoformat(),
                "notional_eur": transaction.notional_eur,
                "fee_eur": transaction.fee_eur,
            }
            for execution in result.executions
            for transaction in execution.rebalance.transactions
        ],
        columns=["execution_date", "notional_eur", "fee_eur"],
    )


def _financial_metrics(result: BacktestResult) -> dict[str, Any]:
    nav = result.nav_series.copy()
    nav.index = pd.DatetimeIndex(pd.to_datetime(nav.index))
    metrics = calculate_metrics(nav, transactions=_transaction_frame(result))
    return {
        key: metrics[key]
        for key in (
            "start_date",
            "end_date",
            "observation_count",
            "cagr",
            "total_return",
            "maximum_drawdown",
            "sharpe_ratio",
            "annualized_volatility",
            "total_fees_eur",
            "total_traded_notional_eur",
            "turnover",
            "final_nav_eur",
        )
    }


def _backtest_config(
    config: MLPTrainingConfig,
    *,
    start_date: date,
    end_date: date,
    policy: str,
    strategy_name: str,
    signal_end_date: date | None = None,
) -> BacktestConfig:
    return BacktestConfig(
        initial_capital_eur=config.initial_capital_eur,
        transaction_cost_bps=config.transaction_cost_bps,
        availability_lag_days=7,
        target_weights={"EUR": 1.0},
        start_date=start_date,
        end_date=end_date,
        signal_end_date=signal_end_date,
        strategy_name=strategy_name,
        rebalance_policy=policy,
    )


def _behavior(weights: np.ndarray) -> dict[str, Any]:
    if weights.ndim != 2 or weights.shape[1] != len(OUTPUT_CURRENCIES):
        raise MLPTrainingError("Allocation behavior analysis requires a time-by-eight matrix")
    differences = np.abs(np.diff(weights, axis=0))
    mean_turnover = float(differences.sum(axis=1).mean()) if len(differences) else 0.0
    max_allocation = float(weights.max(axis=1).max())
    mean_weights = weights.mean(axis=0)
    return {
        "mean_allocation": {
            currency: float(mean_weights[index])
            for index, currency in enumerate(OUTPUT_CURRENCIES)
        },
        "maximum_single_currency_weight": max_allocation,
        "mean_active_foreign_currencies": float((weights[:, 1:] > 0.01).sum(axis=1).mean()),
        "mean_weekly_allocation_l1_change": mean_turnover,
        "max_weekly_allocation_l1_change": float(differences.sum(axis=1).max()) if len(differences) else 0.0,
        "mean_eur_weight": float(mean_weights[0]),
        "mostly_eur": bool(mean_weights[0] >= 0.95),
        "single_currency_concentration": bool(max_allocation >= 0.95),
        "near_fixed_allocations": bool(mean_turnover < 0.01),
        "excessive_allocation_changes": bool(mean_turnover > 1.0),
    }


def _sensitivity(model: AllocationMLP, features: pd.DataFrame, device: torch.device) -> float:
    x = torch.as_tensor(features.to_numpy(dtype=np.float32), dtype=torch.float32, device=device)
    with torch.no_grad():
        baseline = model(x)
        perturbed = model(x + 0.01)
    return float(torch.abs(perturbed - baseline).sum(dim=1).mean().item())


def _run_exact_backtests(
    trained: Mapping[str, Any],
    config: MLPTrainingConfig,
    development_data: MarketData,
) -> dict[str, Any]:
    fold: LoadedFold = trained["fold"]
    validation = fold.validation_frame
    first_signal = validation["signal_date"].min().date()
    last_signal = validation["signal_date"].max().date()
    last_execution = validation["next_execution_date"].max().date()
    development_end = max(development_data.dates)
    if last_execution > development_end:
        raise MLPTrainingError("Validation execution date reaches the protected final-test period")
    prior_dates = [market_date for market_date in development_data.dates if market_date < first_signal]
    if not prior_dates:
        raise MLPTrainingError("No pre-signal market observation is available for initial NAV")
    valuation_start = prior_dates[-1]
    device = torch.device(config.device)
    provider = MLPWeightProvider(
        trained["model"],
        fold.scaler,
        fold.feature_columns,
        device=device,
    )
    run_config = _backtest_config(
        config,
        start_date=valuation_start,
        end_date=last_execution,
        signal_end_date=last_signal,
        policy="weekly",
        strategy_name=f"mlp_v1_fold_{fold.fold_number}",
    )
    result = run_backtest(development_data, provider, run_config)
    expected_signals = tuple(signal.date() for signal in validation["signal_date"])
    expected_executions = tuple(execution.date() for execution in validation["execution_date"])
    actual_signals = tuple(item.signal_date for item in result.executions)
    actual_executions = tuple(item.execution_date for item in result.executions)
    if actual_signals != expected_signals or actual_executions != expected_executions:
        raise MLPTrainingError(
            f"Fold {fold.fold_number} exact backtest dates differ from dataset alignment"
        )
    common_valuation_dates = tuple(record.market_date for record in result.daily_records)
    backtests: dict[str, Any] = {"mlp_v1": result}
    for benchmark in BENCHMARKS:
        policy = "initial_only" if benchmark == "buy_hold_equal_weight" else "weekly"
        benchmark_provider = create_benchmark_provider(benchmark)
        benchmark_config = _backtest_config(
            config,
            start_date=valuation_start,
            end_date=last_execution,
            signal_end_date=last_signal,
            policy=policy,
            strategy_name=benchmark,
        )
        benchmark_result = run_backtest(
            development_data,
            benchmark_provider,
            benchmark_config,
        )
        if tuple(record.market_date for record in benchmark_result.daily_records) != common_valuation_dates:
            raise MLPTrainingError(f"Benchmark {benchmark} valuation dates differ from the MLP")
        benchmark_signals = tuple(item.signal_date for item in benchmark_result.decisions)
        if benchmark_signals != expected_signals:
            raise MLPTrainingError(f"Benchmark {benchmark} signal dates differ from the MLP")
        if benchmark == "buy_hold_equal_weight":
            if tuple(item.execution_date for item in benchmark_result.executions) != expected_executions[:1]:
                raise MLPTrainingError("Buy & Hold must execute only its initial investment")
        elif tuple(item.execution_date for item in benchmark_result.executions) != expected_executions:
            raise MLPTrainingError(f"Benchmark {benchmark} execution dates differ from the MLP")
        backtests[benchmark] = benchmark_result
    return {
        name: {
            "result": item,
            "metrics": _financial_metrics(item),
        }
        for name, item in backtests.items()
    }


def _plot_fold(
    fold_result: Mapping[str, Any],
    *,
    output_dir: Path,
) -> dict[str, Path]:
    fold: LoadedFold = fold_result["fold"]
    fold_number = fold.fold_number
    output_dir.mkdir(parents=True, exist_ok=True)
    paths = {
        "loss": output_dir / f"mlp_v1_fold_{fold_number}_loss.png",
        "nav": output_dir / f"mlp_v1_fold_{fold_number}_nav.png",
        "allocations": output_dir / f"mlp_v1_fold_{fold_number}_allocations.png",
    }
    history = pd.DataFrame(fold_result["history"])
    figure, axis = plt.subplots(figsize=(9, 4.5))
    axis.plot(history["epoch"], history["train_loss"], label="train")
    axis.plot(history["epoch"], history["validation_loss"], label="validation")
    axis.set(title=f"MLP fold {fold_number} — perte", xlabel="Époque", ylabel="Perte de croissance")
    axis.grid(True, alpha=0.25)
    axis.legend()
    figure.tight_layout()
    figure.savefig(paths["loss"], dpi=140)
    plt.close(figure)

    figure, axis = plt.subplots(figsize=(10, 5))
    for name, details in fold_result["backtests"].items():
        series = details["result"].nav_series
        axis.plot(pd.to_datetime(series.index), series.to_numpy(), label=name, linewidth=1.0)
    axis.set(title=f"Fold {fold_number} — NAV de validation (EUR)", xlabel="Date", ylabel="NAV EUR")
    axis.grid(True, alpha=0.25)
    axis.legend()
    figure.tight_layout()
    figure.savefig(paths["nav"], dpi=140)
    plt.close(figure)

    allocations = fold_result["validation_weights"]
    dates = fold.validation_frame["signal_date"]
    figure, axis = plt.subplots(figsize=(11, 5))
    axis.stackplot(
        pd.to_datetime(dates),
        allocations.T,
        labels=OUTPUT_CURRENCIES,
        alpha=0.85,
    )
    axis.set(title=f"Fold {fold_number} — allocations MLP", xlabel="Signal vendredi", ylabel="Poids")
    axis.set_ylim(0, 1)
    axis.grid(True, alpha=0.2)
    axis.legend(loc="upper left", ncol=4, fontsize=8)
    figure.tight_layout()
    figure.savefig(paths["allocations"], dpi=140)
    plt.close(figure)
    return paths


def _fmt(value: Any, *, percentage: bool = False, money: bool = False, multiple: bool = False) -> str:
    if value is None or (isinstance(value, float) and not math.isfinite(value)):
        return "N/D"
    if percentage:
        return f"{value:.2%}"
    if money:
        return f"{value:,.2f} EUR"
    if multiple:
        return f"{value:.3f}×"
    return f"{value:.4f}"


def _render_report(
    config: MLPTrainingConfig,
    manifest: Mapping[str, Any],
    folds: Sequence[Mapping[str, Any]],
    chart_paths: Mapping[str, Mapping[str, Path]],
) -> str:
    lines = [
        "# MLP V1 — validation chronologique",
        "",
        "## Périmètre et garde-fous",
        "",
        (
            "Entraînement et validation sur les seuls folds de développement 8A. Aucune donnée "
            "du test final n'a été utilisée pour l'entraînement, l'arrêt anticipé, les métriques "
            "ou la sélection. Aucun tuning automatique ni trading réel."
        ),
        f"- Frontière du développement : **{manifest['development_end']}** incluse.",
        f"- Test final exclu à partir du **{manifest['test_start_excluded']}**.",
        f"- Dataset SHA-256 : `{manifest['content_sha256']}`.",
        (
            f"- Sources historiques SHA-256 : EUR `{manifest['input_sources']['eur_sha256']}`, "
            f"features `{manifest['input_sources']['feature_sha256']}`."
        ),
        "",
        "## Réseau et entraînement",
        "",
        (
            f"- Architecture : {' → '.join(str(size) for size in config.layer_sizes)}, ReLU sur "
            "les couches cachées, Softmax et sorties "
            f"{', '.join(OUTPUT_CURRENCIES)}."
        ),
        (
            f"- Optimiseur : {config.optimizer}, learning rate {config.learning_rate:g}, "
            f"weight decay {config.weight_decay:g}; maximum {config.max_epochs} époques, "
            f"patience {config.early_stopping_patience}; seed de base {config.random_seed}."
        ),
        (
            f"- Frais entraînement/validation : {config.transaction_cost_bps:g} bps. "
            f"Pénalité drawdown : {config.drawdown_penalty:g} (seuil "
            f"{config.drawdown_threshold:.0%})."
        ),
        (
            "- Perte différentiable : `-mean(log((1 - fee_rate × foreign_weight_L1_turnover) "
            "× (1 + target_weights · asset_returns)))`; le portefeuille pré-trade dérive après "
            "chaque rendement, avec EUR initial à 100 %. Cette estimation ne remplace pas "
            "les frais comptables 6A."
        ),
        (
            "- Early stopping basé exclusivement sur la perte de validation du fold courant. "
            "Les métriques de performance ci-dessous proviennent uniquement du moteur exact 6A/6B."
        ),
        "- Comparateurs du même fold : EUR Cash, Equal Weight Weekly et Buy & Hold Equal Weight.",
        "",
    ]
    metric_keys = (
        ("cagr", "CAGR", "percentage"),
        ("total_return", "Total return", "percentage"),
        ("maximum_drawdown", "Max drawdown", "percentage"),
        ("sharpe_ratio", "Sharpe", "number"),
        ("annualized_volatility", "Volatilité", "percentage"),
        ("total_fees_eur", "Frais", "money"),
        ("turnover", "Turnover", "multiple"),
        ("final_nav_eur", "NAV finale", "money"),
    )
    for fold_result in folds:
        fold_data: LoadedFold = fold_result["fold"]
        number = fold_data.fold_number
        lines.extend(
            [
                f"## Fold {number}",
                "",
                (
                    f"- Entraînement : {fold_data.train_frame['signal_date'].min().date()} à "
                    f"{fold_data.train_frame['signal_date'].max().date()} "
                    f"({len(fold_data.train_frame)} exemples); dernier label "
                    f"{fold_data.train_frame['next_execution_date'].max().date()}."
                ),
                (
                    f"- Validation : {fold_data.validation_frame['signal_date'].min().date()} à "
                    f"{fold_data.validation_frame['signal_date'].max().date()} "
                    f"({len(fold_data.validation_frame)} exemples); exécutions "
                    f"{fold_data.validation_frame['execution_date'].min().date()} à "
                    f"{fold_data.validation_frame['next_execution_date'].max().date()}."
                ),
                f"- Purge : {fold_data.purged_training_examples} exemples.",
                (
                    f"- Époque retenue : {fold_result['best_epoch']}/{len(fold_result['history'])}; "
                    f"perte train du meilleur modèle {fold_result['train_loss_best_model']:.8f}; "
                    f"perte validation {fold_result['validation_loss_best_model']:.8f}."
                ),
                (
                    "- Diagnostic de convergence : perte train réduite "
                    f"{'oui' if fold_result['training_loss_decreased'] else 'non'}; "
                    "perte validation finale supérieure au meilleur point "
                    f"{'oui' if fold_result['validation_loss_worsened_after_best'] else 'non'}."
                ),
                f"- Checkpoint : `{fold_result['checkpoint_path']}`.",
                "",
                "| Stratégie | " + " | ".join(label for _, label, _ in metric_keys) + " |",
                "|---|" + "---:|" * len(metric_keys),
            ]
        )
        for strategy_name, detail in fold_result["backtests"].items():
            metrics = detail["metrics"]
            rendered = []
            for key, _, style in metric_keys:
                rendered.append(
                    _fmt(
                        metrics.get(key),
                        percentage=style == "percentage",
                        money=style == "money",
                        multiple=style == "multiple",
                    )
                )
            lines.append(f"| {strategy_name} | " + " | ".join(rendered) + " |")
        lines.extend(
            [
                "",
                "### Comportement MLP",
                "",
                "| Mesure | Entraînement | Validation |",
                "|---|---:|---:|",
            ]
        )
        for label, key in (
            ("Allocation EUR moyenne", "mean_eur_weight"),
            ("Concentration maximale", "maximum_single_currency_weight"),
            ("Nombre moyen de devises étrangères actives (>1%)", "mean_active_foreign_currencies"),
            ("Variation allocation L1 hebdomadaire", "mean_weekly_allocation_l1_change"),
        ):
            lines.append(
                f"| {label} | "
                f"{_fmt(fold_result['train_behavior'][key], percentage=key in {'mean_eur_weight', 'maximum_single_currency_weight'})} | "
                f"{_fmt(fold_result['validation_behavior'][key], percentage=key in {'mean_eur_weight', 'maximum_single_currency_weight'})} |"
            )
        lines.extend(
            [
                (
                    f"- Sensibilité moyenne L1 à une perturbation +0,01 de chaque feature normalisée : "
                    f"{fold_result['input_sensitivity_l1']:.6f}."
                ),
                "- Indicateurs à examiner : "
                + "; ".join(
                    f"{label}={'oui' if fold_result['validation_behavior'][key] else 'non'}"
                    for label, key in (
                        ("majoritairement EUR", "mostly_eur"),
                        ("concentration mono-devise", "single_currency_concentration"),
                        ("allocations quasi fixes", "near_fixed_allocations"),
                        ("variations excessives", "excessive_allocation_changes"),
                    )
                )
                + ".",
                "",
                "### Graphiques",
                "",
            ]
        )
        for name, path in chart_paths[f"fold_{number}"].items():
            lines.append(f"- {name}: `./{path.name}`")
        lines.append("")

    lines.extend(
        [
            "## Synthèse des trois folds",
            "",
            (
                "La moyenne ci-dessous est la moyenne arithmétique des métriques fold par fold; "
                "elle n'est pas une simulation chaînée entre folds."
            ),
            "",
            "| Stratégie | CAGR moyen | Rendement moyen | Drawdown moyen | Frais moyens | Turnover moyen | NAV finale moyenne |",
            "|---|" + "---:|" * 6,
        ]
    )
    strategy_names = ("mlp_v1", *BENCHMARKS)
    for strategy in strategy_names:
        fold_metrics = [
            item["backtests"][strategy]["metrics"]
            for item in folds
        ]
        avg = {
            key: float(np.mean([metrics[key] for metrics in fold_metrics if metrics[key] is not None]))
            if any(metrics[key] is not None for metrics in fold_metrics)
            else None
            for key in (
                "cagr",
                "total_return",
                "maximum_drawdown",
                "total_fees_eur",
                "turnover",
                "final_nav_eur",
            )
        }
        lines.append(
            f"| {strategy} | {_fmt(avg['cagr'], percentage=True)} | "
            f"{_fmt(avg['total_return'], percentage=True)} | "
            f"{_fmt(avg['maximum_drawdown'], percentage=True)} | "
            f"{_fmt(avg['total_fees_eur'], money=True)} | "
            f"{_fmt(avg['turnover'], multiple=True)} | "
            f"{_fmt(avg['final_nav_eur'], money=True)} |"
        )
    lines.extend(
        [
            "",
            "## Limites et interprétation",
            "",
            (
                "Le coût d'entraînement est une approximation différentiable : le moteur exact "
                "applique les frais par transaction et rééquilibre sur NAV après frais. Les écarts "
                "sont attendus et ne doivent pas être masqués. L'optimisation d'une courte séquence "
                "sur objectifs de croissance ne garantit ni généralisation ni drawdown inférieur "
                "à 20 %. Les taux FRED sont indicatifs, leur disponibilité réelle est inconnue, et "
                "ils ne sont pas des prix exécutables certifiés. Aucun spread, slippage, intérêt ou "
                "carry n'est modélisé."
            ),
            "",
        ]
    )
    return "\n".join(lines)


def run_mlp_experiment(config: MLPTrainingConfig) -> dict[str, Any]:
    """Train three independent fold models and compare exact validation backtests."""
    _, _, manifest = _unique_dataset_artifacts(config.dataset_dir)
    if manifest.get("test_start_excluded") <= manifest.get("development_end"):
        raise MLPTrainingError("The dataset manifest has an invalid development/test boundary")
    market_data = load_market_data(config.eur_data_dir, config.feature_data_dir)
    development_data = _slice_market_data(
        market_data,
        date.fromisoformat(manifest["development_end"]),
    )
    if any(market_date > date.fromisoformat(manifest["development_end"]) for market_date in development_data.dates):
        raise MLPTrainingError("Development data slice contains final-test observations")
    source_hashes = {
        "eur_sha256": development_data.eur_source_sha256,
        "feature_sha256": development_data.feature_source_sha256,
    }
    if (
        source_hashes["eur_sha256"] != manifest["input_sources"]["eur_sha256"]
        or source_hashes["feature_sha256"] != manifest["input_sources"]["feature_sha256"]
    ):
        raise MLPTrainingError("8A dataset input hashes do not match the selected local histories")

    outputs: list[dict[str, Any]] = []
    chart_paths: dict[str, dict[str, Path]] = {}
    report_dir = config.report_path.parent
    for fold_number in range(1, 4):
        fold = _load_fold(config.dataset_dir, manifest, fold_number)
        trained = _train_fold(
            fold,
            config,
            dataset_sha256=manifest["content_sha256"],
            source_hashes=source_hashes,
        )
        trained["fold"] = fold
        with torch.no_grad():
            train_x = _tensor(fold.train_frame, fold.feature_columns, torch.device(config.device))
            train_weights = trained["model"](train_x).cpu().numpy()
        trained["train_weights"] = train_weights
        trained["train_behavior"] = _behavior(train_weights)
        trained["validation_behavior"] = _behavior(trained["validation_weights"])
        trained["input_sensitivity_l1"] = _sensitivity(
            trained["model"],
            fold.validation_frame.loc[:, fold.feature_columns],
            torch.device(config.device),
        )
        trained["backtests"] = _run_exact_backtests(trained, config, development_data)
        outputs.append(trained)
        chart_paths[f"fold_{fold_number}"] = _plot_fold(trained, output_dir=report_dir)

    report_dir.mkdir(parents=True, exist_ok=True)
    report_content = _render_report(config, manifest, outputs, chart_paths)
    config.report_path.write_text(report_content, encoding="utf-8")
    payload = {
        "dataset_sha256": manifest["content_sha256"],
        "development_end": manifest["development_end"],
        "test_start_excluded": manifest["test_start_excluded"],
        "configuration": {
            "layer_sizes": list(config.layer_sizes),
            "optimizer": config.optimizer,
            "learning_rate": config.learning_rate,
            "weight_decay": config.weight_decay,
            "max_epochs": config.max_epochs,
            "early_stopping_patience": config.early_stopping_patience,
            "random_seed": config.random_seed,
            "transaction_cost_bps": config.transaction_cost_bps,
            "device": config.device,
        },
        "folds": [
            {
                "fold_number": item["fold"].fold_number,
                "checkpoint": str(item["checkpoint_path"]),
                "model_manifest": str(item["manifest_path"]),
                "best_epoch": item["best_epoch"],
                "best_validation_loss": item["best_validation_loss"],
                "train_loss_best_model": item["train_loss_best_model"],
                "validation_loss_best_model": item["validation_loss_best_model"],
                "training_loss_decreased": item["training_loss_decreased"],
                "validation_loss_worsened_after_best": item["validation_loss_worsened_after_best"],
                "train_behavior": item["train_behavior"],
                "validation_behavior": item["validation_behavior"],
                "input_sensitivity_l1": item["input_sensitivity_l1"],
                "metrics": {
                    name: details["metrics"]
                    for name, details in item["backtests"].items()
                },
                "charts": {name: path.name for name, path in chart_paths[f"fold_{item['fold'].fold_number}"].items()},
            }
            for item in outputs
        ],
        "final_test_period_evaluated": False,
    }
    json_path = config.report_path.with_suffix(".json")
    json_path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    return {"report_path": config.report_path, "json_path": json_path, "folds": outputs, "charts": chart_paths}


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=PROJECT_ROOT / "configs" / "mlp_v1.yaml")
    args = parser.parse_args(argv)
    config = load_mlp_training_config(args.config)
    outputs = run_mlp_experiment(config)
    print(f"Validation report: {outputs['report_path']}")
    print(f"Validation metrics: {outputs['json_path']}")
    for fold in outputs["folds"]:
        print(
            f"Fold {fold['fold'].fold_number}: best epoch {fold['best_epoch']}, "
            f"checkpoint {fold['checkpoint_path']}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
