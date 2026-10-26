"""PyTorch allocation MLP and differentiable net-growth loss for step 8B."""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
from torch import Tensor, nn

OUTPUT_CURRENCIES = ("EUR", "USD", "GBP", "JPY", "CHF", "CAD", "AUD", "NZD")
DEFAULT_LAYER_SIZES = (91, 32, 16, 8)


@dataclass(frozen=True, slots=True)
class MLPConfig:
    """Configurable dense allocation model topology."""

    layer_sizes: tuple[int, ...] = DEFAULT_LAYER_SIZES

    def __post_init__(self) -> None:
        if (
            len(self.layer_sizes) < 3
            or any(isinstance(size, bool) or not isinstance(size, int) or size <= 0 for size in self.layer_sizes)
            or self.layer_sizes[0] != 91
            or self.layer_sizes[-1] != len(OUTPUT_CURRENCIES)
        ):
            raise TypeError("layer_sizes must start at 91, end at 8, and contain positive widths")


class AllocationMLP(nn.Module):
    """Fully connected ReLU network producing long-only portfolio weights."""

    def __init__(self, config: MLPConfig | None = None) -> None:
        super().__init__()
        config = config or MLPConfig()
        layers: list[nn.Module] = []
        layer_count = len(config.layer_sizes) - 1
        for layer_index, (input_size, output_size) in enumerate(
            zip(config.layer_sizes[:-1], config.layer_sizes[1:], strict=True)
        ):
            layers.append(nn.Linear(input_size, output_size))
            if layer_index < layer_count - 1:
                layers.append(nn.ReLU())
        self.network = nn.Sequential(*layers)
        self.config = config

    def forward(self, features: Tensor) -> Tensor:
        if features.ndim != 2 or features.shape[1] != self.config.layer_sizes[0]:
            raise ValueError(f"features must have shape (batch, {self.config.layer_sizes[0]})")
        return torch.softmax(self.network(features), dim=-1)


@dataclass(frozen=True, slots=True)
class GrowthLossResult:
    """Differentiable portfolio growth diagnostics for a chronological sequence."""

    loss: Tensor
    net_growth_factors: Tensor
    cost_fractions: Tensor
    pretrade_weights: Tensor
    foreign_turnover: Tensor
    maximum_drawdown: Tensor


def differentiable_growth_loss(
    target_weights: Tensor,
    asset_returns: Tensor,
    *,
    fee_rate: float = 0.001,
    drawdown_penalty: float = 0.0,
    drawdown_threshold: float = 0.20,
    lambda_turnover: float = 0.0,
) -> GrowthLossResult:
    """Compute negative mean log net growth with sequential allocation drift.

    Costs are approximated as ``fee_rate * sum(abs(target-pretrade))`` over
    foreign-currency weights. The first pre-trade portfolio is 100% EUR. For
    later periods holdings are drifted by the preceding period's asset returns.
    The period factor applies the fee reduction once, then the target-weight
    gross return once. This remains fully differentiable with respect to weights.
    """
    if target_weights.ndim != 2 or asset_returns.shape != target_weights.shape:
        raise ValueError("target_weights and asset_returns must have the same (time, 8) shape")
    if target_weights.shape[0] == 0 or target_weights.shape[1] != len(OUTPUT_CURRENCIES):
        raise ValueError("the sequence must contain at least one row and exactly eight currencies")
    if not math.isfinite(fee_rate) or not 0 <= fee_rate < 1:
        raise ValueError("fee_rate must be finite and in [0, 1)")
    if not math.isfinite(drawdown_penalty) or drawdown_penalty < 0:
        raise ValueError("drawdown_penalty must be finite and non-negative")
    if not math.isfinite(drawdown_threshold) or not 0 < drawdown_threshold < 1:
        raise ValueError("drawdown_threshold must be finite and in (0, 1)")
    if not math.isfinite(lambda_turnover) or lambda_turnover < 0:
        raise ValueError("lambda_turnover must be finite and non-negative")
    if not bool(torch.isfinite(target_weights).all()) or not bool(torch.isfinite(asset_returns).all()):
        raise ValueError("weights and returns must be finite")
    if bool((target_weights < 0).any()) or not torch.allclose(
        target_weights.sum(dim=1),
        torch.ones_like(target_weights[:, 0]),
        atol=1e-5,
        rtol=1e-5,
    ):
        raise ValueError("target weights must be non-negative and sum to one")
    if bool((asset_returns <= -1).any()):
        raise ValueError("asset returns must be greater than -100%")

    initial_weights = torch.zeros_like(target_weights[0])
    initial_weights[0] = 1.0
    pretrade_rows: list[Tensor] = []
    factors: list[Tensor] = []
    cost_fractions: list[Tensor] = []
    foreign_turnovers: list[Tensor] = []
    pretrade = initial_weights
    for index in range(target_weights.shape[0]):
        target = target_weights[index]
        foreign_turnover = torch.abs(target[1:] - pretrade[1:]).sum()
        cost_fraction = fee_rate * foreign_turnover
        gross_return = torch.dot(target, asset_returns[index])
        net_factor = (1.0 - cost_fraction) * (1.0 + gross_return)
        if not bool(torch.isfinite(net_factor)) or bool(net_factor <= 0):
            raise ValueError("net growth factor must be finite and positive")
        pretrade_rows.append(pretrade)
        cost_fractions.append(cost_fraction)
        foreign_turnovers.append(foreign_turnover)
        factors.append(net_factor)
        if index + 1 < target_weights.shape[0]:
            post_return_values = target * (1.0 + asset_returns[index])
            post_return_nav = post_return_values.sum()
            if not bool(torch.isfinite(post_return_nav)) or bool(post_return_nav <= 0):
                raise ValueError("post-return portfolio value must be finite and positive")
            pretrade = post_return_values / post_return_nav

    net_growth = torch.stack(factors)
    cumulative = torch.cumprod(net_growth, dim=0)
    peaks = torch.cummax(cumulative, dim=0).values
    drawdowns = cumulative / peaks - 1.0
    maximum_drawdown = -drawdowns.min()
    loss = -torch.log(net_growth).mean()
    turnover_tensor = torch.stack(foreign_turnovers)
    loss = loss + lambda_turnover * turnover_tensor.mean()
    if drawdown_penalty:
        loss = loss + drawdown_penalty * torch.relu(maximum_drawdown - drawdown_threshold)
    return GrowthLossResult(
        loss=loss,
        net_growth_factors=net_growth,
        cost_fractions=torch.stack(cost_fractions),
        pretrade_weights=torch.stack(pretrade_rows),
        foreign_turnover=turnover_tensor,
        maximum_drawdown=maximum_drawdown,
    )
