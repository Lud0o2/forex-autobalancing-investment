"""Train and validate fold-specific allocation MLPs on development data only."""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import date, timedelta
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
from forex_v1.portfolio import (
    create_portfolio,
    mark_to_market,
    rebalance_portfolio,
    value_portfolio,
)
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
    lambda_turnover: float
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
        if not math.isfinite(self.lambda_turnover) or self.lambda_turnover < 0:
            raise ValueError("lambda_turnover must be finite and non-negative")
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


@dataclass(frozen=True, slots=True)
class InnerValidationPartition:
    """Chronological inner split contained in one outer training partition."""

    train_frame: pd.DataFrame
    validation_frame: pd.DataFrame
    scaler: FoldStandardizer
    validation_start: date
    purged_training_examples: int
    outer_train_scaler: FoldStandardizer


INNER_VALIDATION_FRACTION = 0.20
ALLOCATION_CHANGE_TOLERANCE = 0.01
EFFECTIVE_HOLDING_THRESHOLD = 0.01


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
        self.used_feature_dates: list[date] = []

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
                self.used_feature_dates.append(market_date)
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
        lambda_turnover=training.get("lambda_turnover", 0.0),
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
        lambda_turnover=config.lambda_turnover,
    )


def _restore_outer_scaled_features(
    frame: pd.DataFrame,
    feature_columns: Sequence[str],
    scaler: FoldStandardizer,
) -> pd.DataFrame:
    values = frame.loc[:, list(feature_columns)].to_numpy(dtype=np.float64, copy=True)
    restored = values * np.asarray(scaler.scales) + np.asarray(scaler.means)
    if not np.isfinite(restored).all():
        raise MLPTrainingError("Could not restore finite raw features from the outer scaler")
    return pd.DataFrame(restored, columns=feature_columns, index=frame.index)


def _inner_validation_partition(
    fold: LoadedFold,
    *,
    validation_fraction: float = INNER_VALIDATION_FRACTION,
) -> InnerValidationPartition:
    """Split the outer train tail, purge overlapping labels, and fit an inner-only scaler."""
    if not 0 < validation_fraction < 1:
        raise ValueError("validation_fraction must be strictly between zero and one")
    frame = fold.train_frame
    validation_count = max(1, math.ceil(len(frame) * validation_fraction))
    if validation_count >= len(frame):
        raise MLPTrainingError(f"Fold {fold.fold_number} is too small for an inner validation split")
    validation_start = frame.iloc[-validation_count]["signal_date"].date()
    validation = frame.loc[frame["signal_date"].dt.date >= validation_start].copy()
    eligible_train = frame.loc[frame["signal_date"].dt.date < validation_start]
    train = eligible_train.loc[
        eligible_train["next_execution_date"].dt.date < validation_start
    ].copy()
    purged = len(eligible_train) - len(train)
    if train.empty or validation.empty:
        raise MLPTrainingError(f"Fold {fold.fold_number} has no usable inner train/validation split")
    if train["next_execution_date"].max().date() >= validation_start:
        raise MLPTrainingError(f"Fold {fold.fold_number} inner training labels overlap validation")
    if validation["next_execution_date"].max() > frame["next_execution_date"].max():
        raise MLPTrainingError(f"Fold {fold.fold_number} inner validation exceeds outer train labels")

    raw_train = _restore_outer_scaled_features(train, fold.feature_columns, fold.scaler)
    raw_validation = _restore_outer_scaled_features(validation, fold.feature_columns, fold.scaler)
    inner_scaler = FoldStandardizer.fit(raw_train)
    train.loc[:, list(fold.feature_columns)] = inner_scaler.transform(raw_train)
    validation.loc[:, list(fold.feature_columns)] = inner_scaler.transform(raw_validation)
    return InnerValidationPartition(
        train_frame=train,
        validation_frame=validation,
        scaler=inner_scaler,
        validation_start=validation_start,
        purged_training_examples=purged,
        outer_train_scaler=fold.scaler,
    )


def _validate_gradients(model: AllocationMLP, loss: Tensor, fold_number: int) -> None:
    if not bool(torch.isfinite(loss)):
        raise MLPTrainingError(f"Fold {fold_number} produced a non-finite training loss")
    loss.backward()
    gradients = [parameter.grad for parameter in model.parameters() if parameter.grad is not None]
    if not gradients or not all(bool(torch.isfinite(gradient).all()) for gradient in gradients):
        raise MLPTrainingError(f"Fold {fold_number} gradients are missing or non-finite")
    if not any(bool((gradient != 0).any()) for gradient in gradients):
        raise MLPTrainingError(f"Fold {fold_number} gradients are all zero")


def _select_inner_epoch(
    partition: InnerValidationPartition,
    fold: LoadedFold,
    config: MLPTrainingConfig,
    *,
    seed: int,
) -> tuple[int, float, list[dict[str, float]]]:
    """Choose a fixed training duration using only the inner chronological split."""
    device = torch.device(config.device)
    _seed_everything(seed)
    model = AllocationMLP(config.model).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )
    x_train = _tensor(partition.train_frame, fold.feature_columns, device)
    y_train = _tensor(partition.train_frame, fold.target_columns, device)
    x_validation = _tensor(partition.validation_frame, fold.feature_columns, device)
    y_validation = _tensor(partition.validation_frame, fold.target_columns, device)
    history: list[dict[str, float]] = []
    best_loss = math.inf
    best_epoch = 0
    stale_epochs = 0
    for epoch in range(1, config.max_epochs + 1):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        _validate_gradients(model, _loss(model, x_train, y_train, config).loss, fold.fold_number)
        torch.nn.utils.clip_grad_norm_(model.parameters(), config.gradient_clip_norm)
        optimizer.step()
        model.eval()
        with torch.no_grad():
            train_loss = float(_loss(model, x_train, y_train, config).loss.item())
            validation_loss = float(_loss(model, x_validation, y_validation, config).loss.item())
        if not math.isfinite(train_loss) or not math.isfinite(validation_loss):
            raise MLPTrainingError(f"Fold {fold.fold_number} produced non-finite inner metrics")
        history.append(
            {"epoch": epoch, "train_loss": train_loss, "validation_loss": validation_loss}
        )
        if validation_loss < best_loss:
            best_loss = validation_loss
            best_epoch = epoch
            stale_epochs = 0
        else:
            stale_epochs += 1
        if stale_epochs >= config.early_stopping_patience:
            break
    if best_epoch == 0:
        raise MLPTrainingError(f"Fold {fold.fold_number} did not select a valid inner epoch")
    return best_epoch, best_loss, history


def _fit_outer_model(
    fold: LoadedFold,
    config: MLPTrainingConfig,
    *,
    epochs: int,
    seed: int,
) -> AllocationMLP:
    """Reset model and optimizer, then fit the complete outer train for fixed epochs."""
    device = torch.device(config.device)
    _seed_everything(seed)
    model = AllocationMLP(config.model).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )
    x_train = _tensor(fold.train_frame, fold.feature_columns, device)
    y_train = _tensor(fold.train_frame, fold.target_columns, device)
    for _ in range(epochs):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        _validate_gradients(model, _loss(model, x_train, y_train, config).loss, fold.fold_number)
        torch.nn.utils.clip_grad_norm_(model.parameters(), config.gradient_clip_norm)
        optimizer.step()
    model.eval()
    return model


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
    inner = _inner_validation_partition(fold)
    best_epoch, best_validation, history = _select_inner_epoch(
        inner,
        fold,
        config,
        seed=seed,
    )
    model = _fit_outer_model(fold, config, epochs=best_epoch, seed=seed)
    x_train = _tensor(fold.train_frame, fold.feature_columns, device)
    y_train = _tensor(fold.train_frame, fold.target_columns, device)
    x_validation = _tensor(fold.validation_frame, fold.feature_columns, device)
    y_validation = _tensor(fold.validation_frame, fold.target_columns, device)
    best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}

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
            "selected_epochs": best_epoch,
            "outer_train_epochs": best_epoch,
            "inner_validation_loss": best_validation,
            "inner_history": history,
            "selection_scope": "purged chronological inner split of outer training only",
            "inner_validation_fraction": INNER_VALIDATION_FRACTION,
            "inner_validation_start": inner.validation_start.isoformat(),
            "inner_train_last_signal": inner.train_frame["signal_date"].max().date().isoformat(),
            "inner_train_last_label_end": inner.train_frame["next_execution_date"].max().date().isoformat(),
            "inner_purged_training_examples": inner.purged_training_examples,
        },
        "finance": {
            "initial_capital_eur": config.initial_capital_eur,
            "transaction_cost_bps": config.transaction_cost_bps,
            "lambda_turnover": config.lambda_turnover,
            "drawdown_penalty": config.drawdown_penalty,
            "drawdown_threshold": config.drawdown_threshold,
            "loss": "negative mean log approximate net growth",
        },
        "scaler": fold.scaler_payload,
        "inner_scaler": {
            **inner.scaler.to_dict(),
            "fit_scope": "inner training X rows only",
        },
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
        lambda_turnover=config.lambda_turnover,
    )
    validation_approx = differentiable_growth_loss(
        torch.as_tensor(validation_weights),
        torch.as_tensor(validation_returns),
        fee_rate=config.fee_rate,
        drawdown_penalty=config.drawdown_penalty,
        drawdown_threshold=config.drawdown_threshold,
        lambda_turnover=config.lambda_turnover,
    )
    return {
        "fold": fold,
        "model": reloaded_model,
        "checkpoint_path": checkpoint_path,
        "manifest_path": manifest_path,
        "history": history,
        "inner_partition": inner,
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
    transaction_cost_bps: float | None = None,
) -> BacktestConfig:
    return BacktestConfig(
        initial_capital_eur=config.initial_capital_eur,
        transaction_cost_bps=(
            config.transaction_cost_bps
            if transaction_cost_bps is None
            else transaction_cost_bps
        ),
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
    if len(provider.used_feature_dates) != len(expected_signals):
        raise MLPTrainingError("The MLP did not record one admissible feature date per validation signal")
    feature_date_by_signal = dict(zip(expected_signals, provider.used_feature_dates, strict=True))
    for signal_date, feature_date in feature_date_by_signal.items():
        if feature_date > signal_date - timedelta(days=7):
            raise MLPTrainingError(f"Fold {fold.fold_number} used a feature after its 7-day cutoff")
    return {
        "strategies": {
            name: {
                "result": item,
                "metrics": _financial_metrics(item),
            }
            for name, item in backtests.items()
        },
        "feature_date_by_signal": feature_date_by_signal,
    }


def _run_fee_sensitivity(
    config: MLPTrainingConfig,
    market_data: MarketData,
    *,
    checkpoint_path: Path,
    fold: LoadedFold,
    transaction_cost_bps: float,
) -> dict[str, Any]:
    """Replay one fixed 8C allocation model and Equal Weight under one fee rate."""
    first_signal = fold.validation_frame["signal_date"].min().date()
    last_signal = fold.validation_frame["signal_date"].max().date()
    last_execution = fold.validation_frame["next_execution_date"].max().date()
    prior_dates = [value for value in market_data.dates if value < first_signal]
    if not prior_dates:
        raise MLPTrainingError(f"Fold {fold.fold_number} has no initial valuation date")
    valuation_start = prior_dates[-1]
    device = torch.device(config.device)
    model = AllocationMLP(config.model).to(device)
    state = torch.load(checkpoint_path, map_location=device, weights_only=True)
    model.load_state_dict(state)
    model.eval()
    provider = MLPWeightProvider(model, fold.scaler, fold.feature_columns, device=device)
    expected_signals = tuple(value.date() for value in fold.validation_frame["signal_date"])
    expected_executions = tuple(value.date() for value in fold.validation_frame["execution_date"])
    results: dict[str, BacktestResult] = {}
    for name, strategy_provider in (
        ("mlp_v1_8c", provider),
        ("equal_weight_weekly", create_benchmark_provider("equal_weight_weekly")),
    ):
        result = run_backtest(
            market_data,
            strategy_provider,
            _backtest_config(
                config,
                start_date=valuation_start,
                end_date=last_execution,
                signal_end_date=last_signal,
                policy="weekly",
                strategy_name=f"{name}_{transaction_cost_bps:g}bps",
                transaction_cost_bps=transaction_cost_bps,
            ),
        )
        if tuple(item.signal_date for item in result.executions) != expected_signals:
            raise MLPTrainingError(f"{name} diagnostic signal dates differ in fold {fold.fold_number}")
        if tuple(item.execution_date for item in result.executions) != expected_executions:
            raise MLPTrainingError(f"{name} diagnostic execution dates differ in fold {fold.fold_number}")
        results[name] = result

    output: dict[str, Any] = {}
    for name, result in results.items():
        targets = [execution.target_weights for execution in result.executions]
        target_array = np.asarray(
            [[target[currency] for currency in OUTPUT_CURRENCIES] for target in targets],
            dtype=np.float64,
        )
        initial = np.zeros(len(OUTPUT_CURRENCIES), dtype=np.float64)
        initial[0] = 1.0
        target_changes = np.abs(np.diff(np.vstack([initial, target_array]), axis=0)).sum(axis=1)
        transactions = [
            transaction
            for execution in result.executions
            for transaction in execution.rebalance.transactions
        ]
        output[name] = {
            "metrics": _financial_metrics(result),
            "return_under_fee_case": result.total_return,
            "cumulative_fees_eur": result.total_fees_eur,
            "turnover": _financial_metrics(result)["turnover"],
            "transaction_count": len(transactions),
            "effective_rebalances": sum(bool(item.rebalance.transactions) for item in result.executions),
            "decision_count": len(result.decisions),
            "mean_target_allocation": {
                currency: float(target_array[:, index].mean())
                for index, currency in enumerate(OUTPUT_CURRENCIES)
            },
            "mean_eur_allocation": float(target_array[:, 0].mean()),
            "mean_target_allocation_l1_change": float(target_changes.mean()),
            "nav_dates": [value.isoformat() for value in result.nav_series.index],
            "result": result,
        }
    if output["mlp_v1_8c"]["nav_dates"] != output["equal_weight_weekly"]["nav_dates"]:
        raise MLPTrainingError("Fee sensitivity strategies do not share common NAV dates")
    return output


def _run_8c_fee_diagnostics(
    config: MLPTrainingConfig,
    market_data: MarketData,
    *,
    audit_payload: Mapping[str, Any],
) -> list[dict[str, Any]]:
    """Compare the same 8C predictions with exact 10 bps and zero-fee ledgers."""
    diagnostics: list[dict[str, Any]] = []
    for audit_fold in audit_payload["folds"]:
        fold_number = int(audit_fold["fold_number"])
        fold = _load_fold(config.dataset_dir, _unique_dataset_artifacts(config.dataset_dir)[2], fold_number)
        checkpoint = Path(audit_fold["checkpoint"])
        fee_results = {
            str(fee): _run_fee_sensitivity(
                config,
                market_data,
                checkpoint_path=checkpoint,
                fold=fold,
                transaction_cost_bps=fee,
            )
            for fee in (10.0, 0.0)
        }
        diagnostics.append(
            {
                "fold_number": fold_number,
                "checkpoint_8c": str(checkpoint),
                "fees_10_bps": fee_results["10.0"],
                "fees_0_bps": fee_results["0.0"],
            }
        )
    return diagnostics


def _synthetic_accounting_cases(fee_rate: float) -> list[dict[str, Any]]:
    """Reconcile the differentiable approximation with 6A on controlled sequences."""
    usd_half = np.zeros((2, len(OUTPUT_CURRENCIES)), dtype=np.float64)
    usd_half[:, 0] = 0.5
    usd_half[:, 1] = 0.5
    usd_half_returns = np.zeros_like(usd_half)
    usd_half_returns[0, 1] = 0.20

    rotation = np.zeros((2, len(OUTPUT_CURRENCIES)), dtype=np.float64)
    rotation[0, 1] = 1.0
    rotation[1, 3] = 1.0

    static_equal = np.zeros((2, len(OUTPUT_CURRENCIES)), dtype=np.float64)
    static_equal[:, 0] = 0.5
    static_equal[:, 1] = 0.5
    static_returns = np.zeros_like(static_equal)
    static_returns[0, 1] = 0.10
    static_returns[1, 1] = -0.05

    eur_cash = np.zeros((2, len(OUTPUT_CURRENCIES)), dtype=np.float64)
    eur_cash[:, 0] = 1.0
    cases = (
        ("eur_cash", eur_cash, np.zeros_like(eur_cash)),
        ("initial_investment_50pct_usd", usd_half[:1].copy(), np.zeros_like(usd_half[:1])),
        ("unchanged_target_with_market_drift", usd_half, usd_half_returns),
        ("full_usd_to_jpy_rotation", rotation, np.zeros_like(rotation)),
        ("constant_50_50_with_drift", static_equal, static_returns),
    )
    output: list[dict[str, Any]] = []
    for name, targets_array, returns_array in cases:
        targets = torch.as_tensor(targets_array, dtype=torch.float64)
        returns = torch.as_tensor(returns_array, dtype=torch.float64)
        approximate = differentiable_growth_loss(targets, returns, fee_rate=fee_rate)
        approximate_net = float(torch.prod(approximate.net_growth_factors).item() - 1.0)
        gross_factors = 1.0 + np.sum(targets_array * returns_array, axis=1)
        gross_return = float(np.prod(gross_factors) - 1.0)

        prices = dict.fromkeys(OUTPUT_CURRENCIES, 1.0)
        state = create_portfolio(10_000.0, prices)
        fees = 0.0
        traded_notional = 0.0
        cash_before_rebalances: list[float] = []
        for target, period_returns in zip(targets_array, returns_array, strict=True):
            cash_before_rebalances.append(state.cash_eur)
            rebalance = rebalance_portfolio(
                state,
                dict(zip(OUTPUT_CURRENCIES, target, strict=True)),
                fee_rate=fee_rate,
            )
            state = rebalance.state
            fees += rebalance.operation_fees_eur
            traded_notional += sum(
                transaction.notional_eur for transaction in rebalance.transactions
            )
            next_prices = {
                currency: prices[currency] * (1.0 + float(period_returns[index]))
                for index, currency in enumerate(OUTPUT_CURRENCIES)
            }
            next_prices["EUR"] = 1.0
            state = mark_to_market(state, next_prices)
            prices = next_prices
        exact_nav = value_portfolio(state).nav_eur
        exact_net = exact_nav / 10_000.0 - 1.0
        output.append(
            {
                "case": name,
                "gross_theoretical_return": gross_return,
                "approximate_net_return": approximate_net,
                "exact_net_return": exact_net,
                "approximation_minus_exact_bps": (approximate_net - exact_net) * 10_000.0,
                "exact_fees_eur": fees,
                "fee_from_notional_eur": traded_notional * fee_rate,
                "cash_before_second_rebalance_eur": (
                    cash_before_rebalances[1] if len(cash_before_rebalances) > 1 else None
                ),
                "nonnegative_cash": state.cash_eur >= -1e-9,
                "nonnegative_positions": all(quantity >= 0 for quantity in state.units.values()),
            }
        )
    return output


def _allocation_diagnostics(
    fold_result: Mapping[str, Any],
    config: MLPTrainingConfig,
) -> dict[str, Any]:
    result: BacktestResult = fold_result["backtests"]["mlp_v1"]["result"]
    validation = fold_result["fold"].validation_frame
    targets = np.asarray(fold_result["validation_weights"], dtype=np.float64)
    returns = validation.loc[:, list(fold_result["fold"].target_columns)].to_numpy(dtype=np.float64)
    if targets.shape != returns.shape:
        raise MLPTrainingError("Validation allocation and return sequences are not aligned")
    initial_target = np.zeros(len(OUTPUT_CURRENCIES), dtype=np.float64)
    initial_target[0] = 1.0
    target_changes = np.abs(np.diff(np.vstack([initial_target, targets]), axis=0)).sum(axis=1)
    gross_factors = 1.0 + np.sum(targets * returns, axis=1)
    gross_return = float(np.prod(gross_factors) - 1.0)
    approximate = differentiable_growth_loss(
        torch.as_tensor(targets, dtype=torch.float64),
        torch.as_tensor(returns, dtype=torch.float64),
        fee_rate=config.fee_rate,
        drawdown_penalty=config.drawdown_penalty,
        drawdown_threshold=config.drawdown_threshold,
        lambda_turnover=config.lambda_turnover,
    )
    approximate_net = float(torch.prod(approximate.net_growth_factors).item() - 1.0)
    actual_weights = np.asarray(
        [[record.weights[currency] for currency in OUTPUT_CURRENCIES] for record in result.daily_records],
        dtype=np.float64,
    )
    if not np.isfinite(actual_weights).all() or (actual_weights < -1e-12).any():
        raise MLPTrainingError("Exact backtest produced invalid or negative realized weights")
    transaction_frame = _transaction_frame(result)
    nav_by_date = pd.Series(
        [record.nav_eur for record in result.daily_records],
        index=pd.to_datetime([record.market_date for record in result.daily_records]),
    )
    yearly: dict[str, dict[str, float]] = {}
    for year in sorted(set(nav_by_date.index.year)):
        year_nav = nav_by_date.loc[nav_by_date.index.year == year]
        if transaction_frame.empty:
            year_transactions = transaction_frame
        else:
            execution_dates = pd.to_datetime(transaction_frame["execution_date"])
            year_transactions = transaction_frame.loc[execution_dates.dt.year == year]
        notional = float(year_transactions["notional_eur"].sum()) if not year_transactions.empty else 0.0
        fees = float(year_transactions["fee_eur"].sum()) if not year_transactions.empty else 0.0
        yearly[str(year)] = {
            "turnover": notional / float(year_nav.mean()) if float(year_nav.mean()) else 0.0,
            "fees_eur": fees,
        }
    effective_rebalances = sum(
        bool(execution.rebalance.transactions)
        for execution in result.executions
    )
    near_identical = int((target_changes <= ALLOCATION_CHANGE_TOLERANCE).sum())
    actual_allocation_means = actual_weights.mean(axis=0)
    behavior = _behavior(targets)
    return {
        "target_mean_allocation": behavior["mean_allocation"],
        "realized_mean_allocation": {
            currency: float(actual_allocation_means[index])
            for index, currency in enumerate(OUTPUT_CURRENCIES)
        },
        "maximum_single_currency_target_weight": float(targets.max()),
        "mean_currencies_held": float(
            (actual_weights >= EFFECTIVE_HOLDING_THRESHOLD).sum(axis=1).mean()
        ),
        "effective_holding_threshold": EFFECTIVE_HOLDING_THRESHOLD,
        "mean_weekly_target_l1_change": float(target_changes.mean()),
        "near_identical_target_weeks": near_identical,
        "near_identical_target_tolerance_l1": ALLOCATION_CHANGE_TOLERANCE,
        "effective_rebalances": effective_rebalances,
        "cumulative_fees_eur": result.total_fees_eur,
        "cumulative_turnover": fold_result["backtests"]["mlp_v1"]["metrics"]["turnover"],
        "annual_turnover_and_fees": yearly,
        "gross_theoretical_return": gross_return,
        "approximate_net_return": approximate_net,
        "exact_net_return": result.total_return,
        "approximate_minus_exact_return_bps": (approximate_net - result.total_return) * 10_000.0,
        "train_in_sample_approximate_net_return": float(
            torch.prod(
                differentiable_growth_loss(
                    torch.as_tensor(fold_result["train_weights"], dtype=torch.float64),
                    torch.as_tensor(
                        fold_result["fold"].train_frame.loc[
                            :, list(fold_result["fold"].target_columns)
                        ].to_numpy(dtype=np.float64),
                        dtype=torch.float64,
                    ),
                    fee_rate=config.fee_rate,
                    drawdown_penalty=config.drawdown_penalty,
                    drawdown_threshold=config.drawdown_threshold,
                ).net_growth_factors
            ).item()
            - 1.0
        ),
        "mostly_eur": bool(behavior["mean_eur_weight"] >= 0.95),
        "single_currency_concentration": bool(
            behavior["maximum_single_currency_weight"] >= 0.95
        ),
        "near_fixed_allocations": bool(
            behavior["mean_weekly_allocation_l1_change"] < ALLOCATION_CHANGE_TOLERANCE
        ),
        "no_negative_positions": all(
            quantity >= 0
            for record in result.daily_records
            for quantity in record.units.values()
        ),
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


def _plot_audit_fold(
    fold_result: Mapping[str, Any],
    *,
    output_dir: Path,
) -> dict[str, Path]:
    fold: LoadedFold = fold_result["fold"]
    number = fold.fold_number
    output_dir.mkdir(parents=True, exist_ok=True)
    paths = {
        "loss": output_dir / f"mlp_v1_audit_fold_{number}_loss.png",
        "nav": output_dir / f"mlp_v1_audit_fold_{number}_nav.png",
        "allocations": output_dir / f"mlp_v1_audit_fold_{number}_allocations.png",
    }
    history = pd.DataFrame(fold_result["history"])
    figure, axis = plt.subplots(figsize=(9, 4.5))
    axis.plot(history["epoch"], history["train_loss"], label="inner train")
    axis.plot(history["epoch"], history["validation_loss"], label="inner validation")
    axis.set(title=f"Audit MLP fold {number} — perte interne", xlabel="Époque", ylabel="Perte")
    axis.grid(True, alpha=0.25)
    axis.legend()
    figure.tight_layout()
    figure.savefig(paths["loss"], dpi=140)
    plt.close(figure)

    figure, axis = plt.subplots(figsize=(10, 5))
    for name, details in fold_result["backtests"].items():
        series = details["result"].nav_series
        axis.plot(pd.to_datetime(series.index), series.to_numpy(), label=name, linewidth=1.0)
    axis.set(title=f"Audit fold {number} — NAV de validation (EUR)", xlabel="Date", ylabel="NAV EUR")
    axis.grid(True, alpha=0.25)
    axis.legend()
    figure.tight_layout()
    figure.savefig(paths["nav"], dpi=140)
    plt.close(figure)

    figure, axis = plt.subplots(figsize=(11, 5))
    axis.stackplot(
        pd.to_datetime(fold.validation_frame["signal_date"]),
        fold_result["validation_weights"].T,
        labels=OUTPUT_CURRENCIES,
        alpha=0.85,
    )
    axis.set(title=f"Audit fold {number} — allocations MLP", xlabel="Signal vendredi", ylabel="Poids")
    axis.set_ylim(0, 1)
    axis.grid(True, alpha=0.2)
    axis.legend(loc="upper left", ncol=4, fontsize=8)
    figure.tight_layout()
    figure.savefig(paths["allocations"], dpi=140)
    plt.close(figure)
    return paths


def _fold_timeline(
    fold_result: Mapping[str, Any],
    *,
    development_end: str,
    test_start: str,
) -> dict[str, Any]:
    fold: LoadedFold = fold_result["fold"]
    validation = fold.validation_frame
    result: BacktestResult = fold_result["backtests"]["mlp_v1"]["result"]
    timeline: list[dict[str, Any]] = []
    label_end_by_signal = {
        row.signal_date.date(): row.next_execution_date.date()
        for row in validation.loc[:, ["signal_date", "next_execution_date"]].itertuples(index=False)
    }
    for decision in result.decisions:
        feature_date = fold_result["feature_date_by_signal"].get(decision.signal_date)
        if decision.signal_date not in label_end_by_signal:
            continue
        if feature_date is None or decision.information_market_date is None:
            raise MLPTrainingError("Outer validation lacks an auditable feature information date")
        if feature_date > decision.information_cutoff_date:
            raise MLPTrainingError("A validation decision used a feature later than its cutoff")
        if decision.execution_date is None or decision.execution_date <= decision.signal_date:
            raise MLPTrainingError("A validation signal lacks a later simulated execution")
        label_end = label_end_by_signal[decision.signal_date]
        if label_end >= date.fromisoformat(test_start):
            raise MLPTrainingError("An outer validation label reaches the protected test period")
        timeline.append(
            {
                "signal_date": decision.signal_date.isoformat(),
                "feature_market_date": feature_date.isoformat(),
                "information_cutoff_date": decision.information_cutoff_date.isoformat(),
                "information_market_date": decision.information_market_date.isoformat(),
                "execution_date": decision.execution_date.isoformat(),
                "return_label_end_date": label_end.isoformat(),
                "status": decision.status,
            }
        )
    if len(timeline) != len(validation):
        raise MLPTrainingError("Outer validation timeline does not reconcile to its dataset rows")
    inner: InnerValidationPartition = fold_result["inner_partition"]
    if inner.train_frame["next_execution_date"].max().date() >= inner.validation_start:
        raise MLPTrainingError("Inner split failed its purged-label boundary")
    if inner.validation_frame["next_execution_date"].max() > fold.train_frame["next_execution_date"].max():
        raise MLPTrainingError("Inner validation exceeds the outer training label boundary")
    return {
        "outer_train": {
            "first_signal": fold.train_frame["signal_date"].min().date().isoformat(),
            "last_signal": fold.train_frame["signal_date"].max().date().isoformat(),
            "last_label_end": fold.train_frame["next_execution_date"].max().date().isoformat(),
        },
        "inner_train": {
            "first_signal": inner.train_frame["signal_date"].min().date().isoformat(),
            "last_signal": inner.train_frame["signal_date"].max().date().isoformat(),
            "last_label_end": inner.train_frame["next_execution_date"].max().date().isoformat(),
            "observation_count": len(inner.train_frame),
        },
        "inner_validation": {
            "first_signal": inner.validation_frame["signal_date"].min().date().isoformat(),
            "last_signal": inner.validation_frame["signal_date"].max().date().isoformat(),
            "last_label_end": inner.validation_frame["next_execution_date"].max().date().isoformat(),
            "observation_count": len(inner.validation_frame),
        },
        "inner_validation_start": inner.validation_start.isoformat(),
        "purged_inner_training_examples": inner.purged_training_examples,
        "outer_validation": {
            "first_signal": validation["signal_date"].min().date().isoformat(),
            "last_signal": validation["signal_date"].max().date().isoformat(),
            "last_label_end": validation["next_execution_date"].max().date().isoformat(),
            "observation_count": len(validation),
        },
        "first_feature_market_date_used": min(
            fold_result["feature_date_by_signal"].values()
        ).isoformat(),
        "maximum_feature_market_date_used": max(
            fold_result["feature_date_by_signal"].values()
        ).isoformat(),
        "first_execution_date": result.executions[0].execution_date.isoformat(),
        "last_execution_date": result.executions[-1].execution_date.isoformat(),
        "first_valuation_date": result.daily_records[0].market_date.isoformat(),
        "last_valuation_date": result.daily_records[-1].market_date.isoformat(),
        "development_end": development_end,
        "final_test_start_excluded": test_start,
        "decisions": timeline,
    }


def _audit_metric_table(metrics_by_strategy: Mapping[str, Mapping[str, Any]]) -> str:
    headings = (
        ("mlp_v1", "MLP V1"),
        ("eur_cash", "EUR Cash"),
        ("equal_weight_weekly", "Equal Weight Weekly"),
        ("buy_hold_equal_weight", "Buy & Hold Equal Weight"),
    )
    lines = [
        "| Stratégie | CAGR | Rendement total | Max drawdown | Sharpe | Volatilité | Frais (EUR) | Turnover | NAV finale (EUR) |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for key, label in headings:
        metrics = metrics_by_strategy[key]
        lines.append(
            f"| {label} | {_fmt(metrics['cagr'], percentage=True)} | "
            f"{_fmt(metrics['total_return'], percentage=True)} | "
            f"{_fmt(metrics['maximum_drawdown'], percentage=True)} | "
            f"{_fmt(metrics['sharpe_ratio'])} | "
            f"{_fmt(metrics['annualized_volatility'], percentage=True)} | "
            f"{_fmt(metrics['total_fees_eur'], money=True)} | "
            f"{_fmt(metrics['turnover'], multiple=True)} | "
            f"{_fmt(metrics['final_nav_eur'], money=True)} |"
        )
    return "\n".join(lines)


def _render_audit_report(
    config: MLPTrainingConfig,
    manifest: Mapping[str, Any],
    folds: Sequence[Mapping[str, Any]],
    timelines: Mapping[int, Mapping[str, Any]],
    diagnostics: Mapping[int, Mapping[str, Any]],
    synthetic_cases: Sequence[Mapping[str, Any]],
    charts: Mapping[int, Mapping[str, Path]],
    initial_results: Mapping[str, Any],
) -> str:
    lines = [
        "# MLP V1 — audit méthodologique 8C",
        "",
        "## Conclusion de l'audit",
        "",
        (
            "L'étape 8B utilisait la validation externe à chaque époque pour sélectionner le checkpoint. "
            "Les mesures financières externes étaient donc contaminées par la sélection et optimistes. "
            "Les nouveaux modèles choisissent désormais le nombre d'époques sur une validation interne "
            "chronologique purgée, puis sont réinitialisés et réentraînés sur tout le train externe. "
            "La validation externe n'intervient qu'après le gel du modèle."
        ),
        "",
        (
            "Les outer folds ont déjà été consultés dans les rapports 8B et ont influencé des décisions "
            "humaines; ils font partie du développement et ne constituent pas un test vierge."
        ),
        f"- Développement inclus jusqu'au **{manifest['development_end']}**.",
        f"- Test final exclu à partir du **{manifest['test_start_excluded']}**; aucune métrique 8C n'y est calculée.",
        f"- Les lectures EUR/features sont filtrées au {manifest['development_end']}; les empreintes complètes ne servent qu'au contrôle d'intégrité des sources.",
        f"- Dataset SHA-256 : `{manifest['content_sha256']}`.",
        (
            f"- Sources historiques SHA-256 : EUR `{manifest['input_sources']['eur_sha256']}`, "
            f"features `{manifest['input_sources']['feature_sha256']}`."
        ),
        "",
        "## Paramètres maintenus",
        "",
        f"- Architecture : `{' → '.join(map(str, config.layer_sizes))}`; optimiseur {config.optimizer}.",
        f"- Learning rate {config.learning_rate:g}; seed de base {config.random_seed}; frais {config.transaction_cost_bps:g} bps.",
        "- Loss inchangée : croissance nette logarithmique différentiable avec dérive et frais approximatifs; aucune recherche d'hyperparamètres.",
        "- Split interne fixé à la dernière tranche chronologique de 20 % du train externe; les exemples train dont le label atteint ou dépasse le signal de séparation sont purgés.",
        "- Scaler interne ajusté uniquement sur les features du inner train. Après sélection, le MLP est réinitialisé et réentraîné sur le train externe complet, transformé avec son scaler externe train-only.",
        "",
        "## Comparaison des performances 8B et audit 8C",
        "",
        "Les résultats 8B sont reproduits depuis l'export précédent; les résultats 8C viennent des nouveaux poids réentraînés. Les CAGR ne sont pas moyennés pour fabriquer un rendement chaîné.",
        "",
        "| Fold | Méthode | Epochs | Rendement total | Max drawdown | NAV finale (EUR) |",
        "|---:|---|---:|---:|---:|---:|",
    ]
    old_folds = initial_results.get("folds", [])
    old_by_fold = {item["fold_number"]: item for item in old_folds}
    for fold in folds:
        number = fold["fold"].fold_number
        old = old_by_fold[number]["metrics"]["mlp_v1"]
        new = fold["backtests"]["mlp_v1"]["metrics"]
        lines.extend(
            [
                f"| {number} | 8B, early stopping externe | — | {_fmt(old['total_return'], percentage=True)} | {_fmt(old['maximum_drawdown'], percentage=True)} | {_fmt(old['final_nav_eur'], money=True)} |",
                f"| {number} | 8C, early stopping interne | {fold['best_epoch']} | {_fmt(new['total_return'], percentage=True)} | {_fmt(new['maximum_drawdown'], percentage=True)} | {_fmt(new['final_nav_eur'], money=True)} |",
            ]
        )
    lines.extend(["", "## Comparaison aux benchmarks par outer fold", ""])
    for fold in folds:
        number = fold["fold"].fold_number
        timeline = timelines[number]
        lines.extend(
            [
                f"### Fold {number}",
                "",
                f"- Outer train : {timeline['outer_train']['first_signal']} à {timeline['outer_train']['last_signal']}; dernier label {timeline['outer_train']['last_label_end']}.",
                f"- Inner train : {timeline['inner_train']['first_signal']} à {timeline['inner_train']['last_signal']}; dernier label {timeline['inner_train']['last_label_end']} ({timeline['inner_train']['observation_count']} exemples).",
                f"- Inner validation : {timeline['inner_validation']['first_signal']} à {timeline['inner_validation']['last_signal']}; dernier label {timeline['inner_validation']['last_label_end']} ({timeline['inner_validation']['observation_count']} exemples).",
                f"- Purge interne : {timeline['purged_inner_training_examples']} exemples; séparation au {timeline['inner_validation_start']}. Epochs choisies : **{fold['best_epoch']}**.",
                f"- Outer validation : signaux {timeline['outer_validation']['first_signal']} à {timeline['outer_validation']['last_signal']}; labels jusqu'au {timeline['outer_validation']['last_label_end']}.",
                f"- Features utilisées : {timeline['first_feature_market_date_used']} à {timeline['maximum_feature_market_date_used']}; exécutions : {timeline['first_execution_date']} à {timeline['last_execution_date']}.",
                f"- Valorisation NAV : {timeline['first_valuation_date']} à {timeline['last_valuation_date']} ({fold['backtests']['mlp_v1']['metrics']['observation_count']} observations).",
                "- Chaque signal est daté un vendredi; feature snapshot ≤ signal − 7 jours; exécution à la première observation après le signal. Le délai est appliqué une seule fois.",
                "",
                _audit_metric_table(
                    {name: details["metrics"] for name, details in fold["backtests"].items()}
                ),
                "",
                "Décisions par signal, date de feature admissible, exécution et fin du label : voir `decisions` dans le JSON.",
                "",
            ]
        )
        diagnostic = diagnostics[number]
        lines.extend(
            [
                "#### Diagnostics d'allocation MLP",
                "",
                "- Poids moyens cibles : " + ", ".join(
                    f"{currency} {weight:.2%}"
                    for currency, weight in diagnostic["target_mean_allocation"].items()
                ) + ".",
                f"- Moyenne de devises effectivement détenues (poids ≥ {EFFECTIVE_HOLDING_THRESHOLD:.0%}) : {diagnostic['mean_currencies_held']:.3f}; poids maximal cible par devise : {diagnostic['maximum_single_currency_target_weight']:.2%}.",
                "- Poids moyens effectivement réalisés : " + ", ".join(
                    f"{currency} {weight:.2%}"
                    for currency, weight in diagnostic["realized_mean_allocation"].items()
                ) + ".",
                f"- Variation cible hebdomadaire L1 moyenne : {diagnostic['mean_weekly_target_l1_change']:.4f}; semaines quasi identiques (L1 ≤ {ALLOCATION_CHANGE_TOLERANCE:.2f}) : {diagnostic['near_identical_target_weeks']}.",
                f"- Rééquilibrages effectifs : {diagnostic['effective_rebalances']}; frais cumulés {diagnostic['cumulative_fees_eur']:.2f} EUR; turnover cumulé {diagnostic['cumulative_turnover']:.3f}×.",
                f"- Rendement brut théorique (sans frais) : {diagnostic['gross_theoretical_return']:.4%}; net approximatif de la loss : {diagnostic['approximate_net_return']:.4%}; net exact 6A : {diagnostic['exact_net_return']:.4%}; écart approx−exact {diagnostic['approximate_minus_exact_return_bps']:.2f} bps.",
                f"- Rendement net approximatif train in-sample : {diagnostic['train_in_sample_approximate_net_return']:.4%}; il n'est pas comparable à une estimation hors échantillon et sert uniquement de diagnostic de surapprentissage.",
                f"- Drapeaux : presque tout EUR={diagnostic['mostly_eur']}; concentration mono-devise={diagnostic['single_currency_concentration']}; allocation quasi fixe={diagnostic['near_fixed_allocations']}; positions non négatives={diagnostic['no_negative_positions']}.",
                "- Turnover annuel et frais annuels : " + "; ".join(
                    f"{year}: {value['turnover']:.3f}×, {value['fees_eur']:.2f} EUR"
                    for year, value in diagnostic["annual_turnover_and_fees"].items()
                ) + ".",
                "",
                "Graphiques : " + ", ".join(f"`./{path.name}`" for path in charts[number].values()) + ".",
                "",
            ]
        )
    lines.extend(
        [
            "## Réconciliation perte approximative / moteur exact",
            "",
            "La loss applique une réduction multiplicative par approximation du turnover; 6A calcule les transactions et frais sur les montants convertis, puis valorise les positions après frais. L'écart n'est pas une anomalie de comptabilité à lui seul.",
            "",
            "| Séquence synthétique | Rendement brut théorique | Net approx. | Net exact 6A | Approx. − exact (bps) | Frais exacts (EUR) |",
            "|---|---:|---:|---:|---:|---:|",
        ]
    )
    for case in synthetic_cases:
        lines.append(
            f"| {case['case']} | {_fmt(case['gross_theoretical_return'], percentage=True)} | "
            f"{_fmt(case['approximate_net_return'], percentage=True)} | "
            f"{_fmt(case['exact_net_return'], percentage=True)} | "
            f"{case['approximation_minus_exact_bps']:.3f} | "
            f"{case['exact_fees_eur']:.4f} |"
        )
    lines.extend(
        [
            "",
            "Les séquences couvrent EUR constant, investissement initial, dérive de poids, rotation USD→JPY et allocation fixe avec variation des prix. La comptabilité 6A n'a pas été modifiée.",
            "",
            "## Période chaînée",
            "",
            "Aucune performance chaînée n'est publiée : chaque fold réinitialise le portefeuille et ses périodes de valorisation se chevauchent aux transitions. Les NAV initiales/finales par fold restent les comparaisons honnêtes; concaténer les rendements naïvement dupliquerait ou omettrait des intervalles.",
            "",
            "## Limites",
            "",
            "Les folds externes ont déjà été consultés lors de l'étape 8B; cette correction retire le biais d'early stopping externe mais ne restaure pas leur statut de données vierges. La disponibilité réelle des observations FRED est inconnue; le délai de sept jours est une hypothèse de recherche, pas un horodatage de publication. Les taux ne sont pas des prix exécutables certifiés; spread, slippage, carry et intérêts restent exclus.",
            "",
        ]
    )
    return "\n".join(lines)


def run_mlp_audit_experiment(config: MLPTrainingConfig) -> dict[str, Any]:
    """Run step 8C with inner-only epoch selection and versioned artifacts."""
    _, _, manifest = _unique_dataset_artifacts(config.dataset_dir)
    if manifest.get("test_start_excluded") <= manifest.get("development_end"):
        raise MLPTrainingError("The dataset manifest has an invalid development/test boundary")
    previous_report_path = config.report_path.parent / "mlp_v1_validation.json"
    try:
        previous_report = json.loads(previous_report_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise MLPTrainingError(f"Cannot read prior 8B report {previous_report_path}: {error}") from error

    development_end = date.fromisoformat(manifest["development_end"])
    development_data = load_market_data(
        config.eur_data_dir,
        config.feature_data_dir,
        end_date=development_end,
    )
    if any(market_date > development_end for market_date in development_data.dates):
        raise MLPTrainingError("Development slice contains final-test observations")
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
    timelines: dict[int, dict[str, Any]] = {}
    diagnostics: dict[int, dict[str, Any]] = {}
    chart_paths: dict[int, dict[str, Path]] = {}
    report_dir = config.report_path.parent
    for fold_number in range(1, 4):
        fold = _load_fold(config.dataset_dir, manifest, fold_number)
        if fold.validation_frame["next_execution_date"].max().date() >= date.fromisoformat(
            manifest["test_start_excluded"]
        ):
            raise MLPTrainingError("Outer validation labels enter the protected final-test period")
        trained = _train_fold(
            fold,
            config,
            dataset_sha256=manifest["content_sha256"],
            source_hashes=source_hashes,
        )
        trained["fold"] = fold
        with torch.no_grad():
            train_x = _tensor(fold.train_frame, fold.feature_columns, torch.device(config.device))
            trained["train_weights"] = trained["model"](train_x).cpu().numpy()
        trained["train_behavior"] = _behavior(trained["train_weights"])
        trained["validation_behavior"] = _behavior(trained["validation_weights"])
        trained["input_sensitivity_l1"] = _sensitivity(
            trained["model"],
            fold.validation_frame.loc[:, fold.feature_columns],
            torch.device(config.device),
        )
        exact_validation = _run_exact_backtests(trained, config, development_data)
        trained["backtests"] = exact_validation["strategies"]
        trained["feature_date_by_signal"] = exact_validation["feature_date_by_signal"]
        reference_nav_dates = tuple(
            record.market_date for record in trained["backtests"]["mlp_v1"]["result"].daily_records
        )
        for strategy, detail in trained["backtests"].items():
            dates = tuple(record.market_date for record in detail["result"].daily_records)
            if dates != reference_nav_dates:
                raise MLPTrainingError(f"{strategy} does not share exact NAV dates in fold {fold_number}")
        trained["diagnostics"] = _allocation_diagnostics(trained, config)
        outputs.append(trained)
        timelines[fold_number] = _fold_timeline(
            trained,
            development_end=manifest["development_end"],
            test_start=manifest["test_start_excluded"],
        )
        diagnostics[fold_number] = trained["diagnostics"]
        chart_paths[fold_number] = _plot_audit_fold(trained, output_dir=report_dir)

    synthetic_cases = _synthetic_accounting_cases(config.fee_rate)
    report_dir.mkdir(parents=True, exist_ok=True)
    config.report_path.write_text(
        _render_audit_report(
            config,
            manifest,
            outputs,
            timelines,
            diagnostics,
            synthetic_cases,
            chart_paths,
            previous_report,
        ),
        encoding="utf-8",
    )
    payload = {
        "audit_version": 1,
        "methodology": {
            "outer_validation_used_for_epoch_selection_in_8b": True,
            "inner_validation_fraction_of_outer_train": INNER_VALIDATION_FRACTION,
            "inner_purge_rule": "training next_execution_date must be strictly before inner validation first signal",
            "inner_scaler_fit_scope": "inner training features only",
            "outer_scaler_fit_scope": "complete outer training features only",
            "outer_validation_used_for_training_or_epoch_selection": False,
            "test_period_evaluated": False,
            "availability_lag_days": 7,
            "availability_lag_applied_once": True,
            "chained_oos_result": None,
            "chained_oos_exclusion_reason": "fold NAV spans overlap and each validation starts with a reset portfolio",
        },
        "dataset_sha256": manifest["content_sha256"],
        "source_history_sha256": source_hashes,
        "baseline_8b_report_sha256": hashlib.sha256(
            previous_report_path.read_bytes()
        ).hexdigest(),
        "development_end": manifest["development_end"],
        "test_start_excluded": manifest["test_start_excluded"],
        "configuration": {
            "layer_sizes": list(config.layer_sizes),
            "optimizer": config.optimizer,
            "learning_rate": config.learning_rate,
            "weight_decay": config.weight_decay,
            "random_seed": config.random_seed,
            "transaction_cost_bps": config.transaction_cost_bps,
            "loss": "unchanged differentiable net-growth loss",
        },
        "synthetic_accounting_comparison": synthetic_cases,
        "folds": [],
    }
    prior_folds = {item["fold_number"]: item for item in previous_report["folds"]}
    for item in outputs:
        number = item["fold"].fold_number
        payload["folds"].append(
            {
                "fold_number": number,
                "checkpoint": str(item["checkpoint_path"]),
                "model_manifest": str(item["manifest_path"]),
                "selected_epochs": item["best_epoch"],
                "inner_best_validation_loss": item["best_validation_loss"],
                "inner_history": item["history"],
                "timeline": timelines[number],
                "initial_8b_mlp_metrics": prior_folds[number]["metrics"]["mlp_v1"],
                "metrics": {
                    name: detail["metrics"]
                    for name, detail in item["backtests"].items()
                },
                "diagnostics": diagnostics[number],
                "charts": {name: path.name for name, path in chart_paths[number].items()},
            }
        )
    json_path = config.report_path.with_suffix(".json")
    json_path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    return {
        "report_path": config.report_path,
        "json_path": json_path,
        "folds": outputs,
        "charts": chart_paths,
        "synthetic_accounting_comparison": synthetic_cases,
    }


def _turnover_result_summary(result: BacktestResult) -> dict[str, Any]:
    executions = result.executions
    targets = np.asarray(
        [
            [execution.target_weights[currency] for currency in OUTPUT_CURRENCIES]
            for execution in executions
        ],
        dtype=np.float64,
    )
    initial = np.zeros(len(OUTPUT_CURRENCIES), dtype=np.float64)
    initial[0] = 1.0
    changes = np.abs(np.diff(np.vstack([initial, targets]), axis=0)).sum(axis=1)
    transactions = [
        transaction
        for execution in executions
        for transaction in execution.rebalance.transactions
    ]
    return {
        "metrics": _financial_metrics(result),
        "transaction_count": len(transactions),
        "effective_rebalances": sum(bool(item.rebalance.transactions) for item in executions),
        "decision_count": len(result.decisions),
        "mean_eur_allocation": float(targets[:, 0].mean()) if len(targets) else 1.0,
        "mean_target_allocation_l1_change": float(changes.mean()) if len(changes) else 0.0,
        "mean_target_allocation": {
            currency: float(targets[:, index].mean()) if len(targets) else float(index == 0)
            for index, currency in enumerate(OUTPUT_CURRENCIES)
        },
    }


def _plot_turnover_comparison(
    folds: Sequence[Mapping[str, Any]],
    diagnostics: Sequence[Mapping[str, Any]],
    output_dir: Path,
) -> dict[str, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    nav_path = output_dir / "mlp_v1_turnover_nav.png"
    turnover_path = output_dir / "mlp_v1_turnover_comparison.png"
    figure, axes = plt.subplots(3, 1, figsize=(12, 14))
    for index, fold in enumerate(folds):
        axis = axes[index]
        number = fold["fold"].fold_number
        baseline = diagnostics[index]["fees_10_bps"]
        axis.plot(
            pd.to_datetime(baseline["mlp_v1_8c"]["result"].nav_series.index),
            baseline["mlp_v1_8c"]["result"].nav_series.to_numpy(),
            label="MLP V1 8C",
        )
        axis.plot(
            pd.to_datetime(fold["backtests"]["mlp_v1"]["result"].nav_series.index),
            fold["backtests"]["mlp_v1"]["result"].nav_series.to_numpy(),
            label="MLP turnover penalty",
        )
        for name in ("equal_weight_weekly", "eur_cash"):
            result: BacktestResult = fold["backtests"][name]["result"]
            axis.plot(pd.to_datetime(result.nav_series.index), result.nav_series.to_numpy(), label=name)
        axis.set_title(f"Fold {number} — NAV à frais exacts de 10 bps")
        axis.set_ylabel("NAV EUR")
        axis.grid(True, alpha=0.25)
        axis.legend()
    axes[-1].set_xlabel("Date")
    figure.tight_layout()
    figure.savefig(nav_path, dpi=140)
    plt.close(figure)

    labels = [f"Fold {fold['fold'].fold_number}" for fold in folds]
    series = {
        "MLP V1 8C": [
            item["fees_10_bps"]["mlp_v1_8c"]["metrics"]["turnover"]
            for item in diagnostics
        ],
        "MLP turnover penalty": [
            fold["backtests"]["mlp_v1"]["metrics"]["turnover"] for fold in folds
        ],
        "Equal Weight Weekly": [
            item["fees_10_bps"]["equal_weight_weekly"]["metrics"]["turnover"]
            for item in diagnostics
        ],
    }
    x = np.arange(len(labels))
    width = 0.24
    figure, axis = plt.subplots(figsize=(9, 5))
    for offset, (name, values) in enumerate(series.items()):
        axis.bar(x + (offset - 1) * width, values, width, label=name)
    axis.set_xticks(x, labels)
    axis.set_ylabel("Cumulative turnover (×)")
    axis.set_title("Turnover cumulé par fold — frais exacts de 10 bps")
    axis.grid(True, axis="y", alpha=0.25)
    axis.legend()
    figure.tight_layout()
    figure.savefig(turnover_path, dpi=140)
    plt.close(figure)
    return {"nav": nav_path, "turnover": turnover_path}


def _render_turnover_report(
    folds: Sequence[Mapping[str, Any]],
    diagnostics: Sequence[Mapping[str, Any]],
    *,
    lambda_turnover: float,
    charts: Mapping[str, Path],
    development_end: str,
    test_start: str,
) -> str:
    lines = [
        "# MLP V1 — expérience de pénalité de turnover (étape 8D)",
        "",
        "## Protocole",
        "",
        "- Baseline A : checkpoints 8C existants, sans réentraînement; pénalité supplémentaire λ = 0.",
        f"- Variante B : même architecture 91 → 32 → 16 → 8, AdamW, learning rate 0.001, seed 42, loss 8C et frais 10 bps; seul changement: `lambda_turnover = {lambda_turnover:g}`.",
        "- Le terme ajouté est `lambda_turnover × mean(sum(abs(w_foreign - q_foreign)))`; `q` représente les poids réellement prétransactionnels dérivés par les rendements précédents. Il inclut l'achat initial depuis EUR et reste différentiable.",
        "- λ est une pénalité de loss, pas un frais de transaction. Le backtester B continue de facturer seulement 10 bps.",
        "- Validation interne chronologique 8C, purge, scaler inner train-only, puis réinitialisation et réentraînement sur tout outer train.",
        f"- Historique limité au développement jusqu'au {development_end}; le test final débute le {test_start} et n'est pas lu pour les expériences.",
        "- Résultats exploratoires uniquement : les folds externes ont déjà été consultés en 8B/8C. λ a été fixé avant cette exécution et n'a pas été ajusté.",
        "",
        "## Impact des frais sur les allocations 8C existantes",
        "",
        "Les mêmes checkpoints 8C et mêmes allocations sont rejoués dans le moteur exact avec 10 bps puis 0 bps. Le rendement à frais nuls mesure l'allocation sans coûts simulés; la différence entre rendements est l'effet composé des frais, distinct de la somme en EUR des transactions.",
        "",
        "| Fold | Stratégie | Rendement à frais nuls (brut simulé) | Rendement net à 10 bps | Écart brut−net | Frais cumulés EUR | Turnover | Transactions | Rééquilibrages | Variation L1 cible moyenne |",
        "|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for diagnostic in diagnostics:
        number = diagnostic["fold_number"]
        for strategy, label in (
            ("mlp_v1_8c", "MLP V1 8C"),
            ("equal_weight_weekly", "Equal Weight Weekly"),
        ):
            gross = diagnostic["fees_0_bps"][strategy]
            net = diagnostic["fees_10_bps"][strategy]
            gross_return = gross["return_under_fee_case"]
            net_return = net["return_under_fee_case"]
            lines.append(
                f"| {number} | {label} | {_fmt(gross_return, percentage=True)} | "
                f"{_fmt(net_return, percentage=True)} | "
                f"{_fmt(gross_return - net_return, percentage=True)} | "
                f"{net['cumulative_fees_eur']:.2f} | {net['turnover']:.3f}× | "
                f"{net['transaction_count']} | {net['effective_rebalances']} | "
                f"{net['mean_target_allocation_l1_change']:.4f} |"
            )
    lines.extend(["", "## Comparaison finale à 10 bps", ""])
    for fold, diagnostic in zip(folds, diagnostics, strict=True):
        number = fold["fold"].fold_number
        lines.extend(
            [
                f"### Fold {number}",
                "",
                "| Stratégie | Rendement net | CAGR | Max drawdown | Sharpe | NAV finale EUR | Frais EUR | Turnover | Transactions | Rééquilibrages | Poids EUR moyen | Variation L1 moyenne |",
                "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
            ]
        )
        candidates = {
            "MLP V1 audit 8C": diagnostic["fees_10_bps"]["mlp_v1_8c"],
            "MLP turnover penalty": _turnover_result_summary(
                fold["backtests"]["mlp_v1"]["result"]
            ),
            "Equal Weight Weekly": diagnostic["fees_10_bps"]["equal_weight_weekly"],
            "EUR Cash": _turnover_result_summary(fold["backtests"]["eur_cash"]["result"]),
        }
        for strategy, summary in candidates.items():
            metrics = summary["metrics"]
            lines.append(
                f"| {strategy} | {_fmt(metrics['total_return'], percentage=True)} | "
                f"{_fmt(metrics['cagr'], percentage=True)} | "
                f"{_fmt(metrics['maximum_drawdown'], percentage=True)} | "
                f"{_fmt(metrics['sharpe_ratio'])} | "
                f"{_fmt(metrics['final_nav_eur'], money=True)} | "
                f"{_fmt(metrics['total_fees_eur'], money=True)} | "
                f"{_fmt(metrics['turnover'], multiple=True)} | "
                f"{summary['transaction_count']} | {summary['effective_rebalances']} | "
                f"{summary['mean_eur_allocation']:.2%} | "
                f"{summary['mean_target_allocation_l1_change']:.4f} |"
            )
        lines.extend(["", ""])

    lines.extend(["## Réponses aux questions", ""])
    for index, (fold, diagnostic) in enumerate(zip(folds, diagnostics, strict=True)):
        number = fold["fold"].fold_number
        gross_mlp = diagnostic["fees_0_bps"]["mlp_v1_8c"]["return_under_fee_case"]
        gross_equal = diagnostic["fees_0_bps"]["equal_weight_weekly"]["return_under_fee_case"]
        net_mlp = diagnostic["fees_10_bps"]["mlp_v1_8c"]["return_under_fee_case"]
        net_equal = diagnostic["fees_10_bps"]["equal_weight_weekly"]["return_under_fee_case"]
        old_turnover = diagnostic["fees_10_bps"]["mlp_v1_8c"]["turnover"]
        new_turnover = fold["backtests"]["mlp_v1"]["metrics"]["turnover"]
        old_fee = diagnostic["fees_10_bps"]["mlp_v1_8c"]["cumulative_fees_eur"]
        new_fee = fold["backtests"]["mlp_v1"]["result"].total_fees_eur
        lines.append(
            f"- Fold {number}: meilleur rendement brut que Equal Weight = "
            f"{'oui' if gross_mlp > gross_equal else 'non'} "
            f"({gross_mlp:.2%} contre {gross_equal:.2%}); impact frais composé MLP "
            f"{gross_mlp - net_mlp:.2%} contre Equal Weight {gross_equal - net_equal:.2%}; "
            f"turnover {old_turnover:.3f}× → {new_turnover:.3f}× "
            f"({(1 - new_turnover / old_turnover) if old_turnover else 0:.1%} de réduction), "
            f"frais {old_fee:.2f} → {new_fee:.2f} EUR."
        )
    lines.extend(
        [
            "",
            "Un turnover plus faible n'est pas, à lui seul, une amélioration financière. Le tableau montre directement si le rendement net et le drawdown évoluent dans le bon sens; aucun choix ultérieur de λ n'est fait sur ces folds.",
            "",
            "## Graphiques",
            "",
            f"- Courbes NAV comparatives : `./{charts['nav'].name}`",
            f"- Turnover cumulé comparatif : `./{charts['turnover'].name}`",
            "",
            "## Limites",
            "",
            "Les cours FRED sont indicatifs, non certifiés exécutables, et leur disponibilité réelle est inconnue. Le délai de sept jours est une convention de recherche. Spread, slippage, intérêt et carry ne sont pas inclus. Les folds ont été précédemment examinés et ne sont pas une validation indépendante de généralisation.",
            "",
        ]
    )
    return "\n".join(lines)


def run_mlp_turnover_experiment(config: MLPTrainingConfig) -> dict[str, Any]:
    """Replay 8C costs and compare one fixed turnover-penalized experiment."""
    if config.lambda_turnover <= 0:
        raise MLPTrainingError("Turnover experiment requires a strictly positive lambda_turnover")
    audit_path = config.report_path.parent / "mlp_v1_audit.json"
    try:
        audit_payload = json.loads(audit_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise MLPTrainingError(f"Cannot read 8C audit results {audit_path}: {error}") from error
    if audit_payload.get("methodology", {}).get("test_period_evaluated") is not False:
        raise MLPTrainingError("The 8C baseline does not certify exclusion of the final test period")
    manifest_path = next(iter(sorted(config.dataset_dir.glob("ml_dataset_v1_*.json"))), None)
    if manifest_path is None:
        raise MLPTrainingError("No immutable 8A dataset manifest is available")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    development_end = date.fromisoformat(manifest["development_end"])
    market_data = load_market_data(
        config.eur_data_dir,
        config.feature_data_dir,
        end_date=development_end,
    )
    if max(market_data.dates) != development_end:
        raise MLPTrainingError("Turnover experiment market data exceed the development boundary")
    if (
        market_data.eur_source_sha256 != manifest["input_sources"]["eur_sha256"]
        or market_data.feature_source_sha256 != manifest["input_sources"]["feature_sha256"]
    ):
        raise MLPTrainingError("Turnover experiment source hashes differ from the 8A manifest")
    fee_diagnostics = _run_8c_fee_diagnostics(
        config,
        market_data,
        audit_payload=audit_payload,
    )
    outputs: list[dict[str, Any]] = []
    for fold_number in range(1, 4):
        fold = _load_fold(config.dataset_dir, manifest, fold_number)
        trained = _train_fold(
            fold,
            config,
            dataset_sha256=manifest["content_sha256"],
            source_hashes={
                "eur_sha256": market_data.eur_source_sha256,
                "feature_sha256": market_data.feature_source_sha256,
            },
        )
        trained["fold"] = fold
        trained["train_weights"] = trained["model"](
            _tensor(fold.train_frame, fold.feature_columns, torch.device(config.device))
        ).detach().cpu().numpy()
        trained["backtests"] = _run_exact_backtests(trained, config, market_data)["strategies"]
        expected_dates = tuple(
            record.market_date for record in trained["backtests"]["mlp_v1"]["result"].daily_records
        )
        for strategy, details in trained["backtests"].items():
            if tuple(record.market_date for record in details["result"].daily_records) != expected_dates:
                raise MLPTrainingError(
                    f"Turnover comparison strategy {strategy} has mismatched dates in fold {fold_number}"
                )
        outputs.append(trained)

    for index, fold in enumerate(outputs):
        baseline_dates = fee_diagnostics[index]["fees_10_bps"]["mlp_v1_8c"]["nav_dates"]
        for strategy, details in fold["backtests"].items():
            current_dates = [
                value.isoformat() for value in details["result"].nav_series.index
            ]
            if current_dates != baseline_dates:
                raise MLPTrainingError(
                    f"8C and turnover comparison NAV dates differ for {strategy}, "
                    f"fold {index + 1}"
                )

    charts = _plot_turnover_comparison(outputs, fee_diagnostics, config.report_path.parent)
    report_content = _render_turnover_report(
        outputs,
        fee_diagnostics,
        lambda_turnover=config.lambda_turnover,
        charts=charts,
        development_end=manifest["development_end"],
        test_start=manifest["test_start_excluded"],
    )
    config.report_path.parent.mkdir(parents=True, exist_ok=True)
    config.report_path.write_text(report_content, encoding="utf-8")

    folds_json = []
    for fold, diagnostic in zip(outputs, fee_diagnostics, strict=True):
        number = fold["fold"].fold_number
        folds_json.append(
            {
                "fold_number": number,
                "selected_epochs": fold["best_epoch"],
                "checkpoint": str(fold["checkpoint_path"]),
                "model_manifest": str(fold["manifest_path"]),
                "fee_sensitivity_8c": {
                    fee_key: {
                        strategy: {
                            key: value
                            for key, value in strategy_result.items()
                            if key != "result"
                        }
                        for strategy, strategy_result in fee_diagnostics[number - 1][fee_key].items()
                    }
                    for fee_key in ("fees_10_bps", "fees_0_bps")
                },
                "metrics": {
                    strategy: detail["metrics"]
                    for strategy, detail in fold["backtests"].items()
                },
                "diagnostics": {
                    "transaction_count": sum(
                        len(execution.rebalance.transactions)
                        for execution in fold["backtests"]["mlp_v1"]["result"].executions
                    ),
                    "effective_rebalances": sum(
                        bool(execution.rebalance.transactions)
                        for execution in fold["backtests"]["mlp_v1"]["result"].executions
                    ),
                    "mean_eur_allocation": _turnover_result_summary(
                        fold["backtests"]["mlp_v1"]["result"]
                    )["mean_eur_allocation"],
                    "mean_target_allocation_l1_change": _turnover_result_summary(
                        fold["backtests"]["mlp_v1"]["result"]
                    )["mean_target_allocation_l1_change"],
                    "mean_target_allocation": _turnover_result_summary(
                        fold["backtests"]["mlp_v1"]["result"]
                    )["mean_target_allocation"],
                },
            }
        )
    payload = {
        "experiment": "mlp_v1_turnover_penalty",
        "lambda_turnover": config.lambda_turnover,
        "regularizer": "lambda_turnover * mean(sum(abs(target_foreign - drifted_pretrade_foreign)))",
        "transaction_cost_bps": config.transaction_cost_bps,
        "architecture": list(config.layer_sizes),
        "optimizer": config.optimizer,
        "learning_rate": config.learning_rate,
        "random_seed": config.random_seed,
        "dataset_sha256": manifest["content_sha256"],
        "source_history_sha256": {
            "eur": market_data.eur_source_sha256,
            "features": market_data.feature_source_sha256,
        },
        "development_end": manifest["development_end"],
        "test_start_excluded": manifest["test_start_excluded"],
        "test_period_evaluated": False,
        "outer_folds_previously_examined": True,
        "charts": {name: path.name for name, path in charts.items()},
        "folds": folds_json,
    }
    json_path = config.report_path.with_suffix(".json")
    json_path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    return {
        "report_path": config.report_path,
        "json_path": json_path,
        "folds": outputs,
        "fee_diagnostics": fee_diagnostics,
        "charts": charts,
    }


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
        exact_validation = _run_exact_backtests(trained, config, development_data)
        trained["backtests"] = exact_validation["strategies"]
        trained["feature_date_by_signal"] = exact_validation["feature_date_by_signal"]
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
    parser.add_argument(
        "--audit",
        action="store_true",
        help="Run the isolated-inner-validation 8C audit instead of the original 8B pipeline",
    )
    parser.add_argument(
        "--turnover",
        action="store_true",
        help="Run the single fixed turnover-regularization experiment from step 8D",
    )
    args = parser.parse_args(argv)
    config = load_mlp_training_config(args.config)
    if args.audit and args.turnover:
        parser.error("--audit and --turnover are mutually exclusive")
    if args.turnover:
        outputs = run_mlp_turnover_experiment(config)
    elif args.audit:
        outputs = run_mlp_audit_experiment(config)
    else:
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
