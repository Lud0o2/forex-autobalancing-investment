"""Static Matplotlib visualizations for performance reports."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.figure import Figure

from forex_v1.evaluation.returns import monthly_return_matrix
from forex_v1.evaluation.risk import drawdown_series


def create_performance_plots(
    nav: pd.Series,
    annual_returns: pd.DataFrame,
    monthly_returns: pd.DataFrame,
    allocations: pd.DataFrame,
    transactions: pd.DataFrame,
    output_dir: str | Path,
) -> dict[str, Path]:
    """Create the six requested plots from the same filtered input records."""

    directory = Path(output_dir)
    directory.mkdir(parents=True, exist_ok=True)
    outputs = {
        "capital": directory / "performance_capital_curve.png",
        "drawdown": directory / "performance_drawdown.png",
        "annual": directory / "performance_annual_returns.png",
        "monthly": directory / "performance_monthly_heatmap.png",
        "weights": directory / "performance_currency_weights.png",
        "fees": directory / "performance_cumulative_fees.png",
    }

    figure, axis = plt.subplots(figsize=(11, 5))
    axis.plot(nav.index, nav.to_numpy(), linewidth=1.1)
    axis.set(title="Capital en EUR", xlabel="Date", ylabel="NAV (EUR)")
    axis.grid(True, alpha=0.25)
    _save(figure, outputs["capital"])

    figure, axis = plt.subplots(figsize=(11, 4))
    underwater = drawdown_series(nav)
    axis.fill_between(underwater.index, underwater.to_numpy() * 100, 0, alpha=0.35)
    axis.plot(underwater.index, underwater.to_numpy() * 100, linewidth=0.9)
    axis.set(title="Drawdown", xlabel="Date", ylabel="Drawdown (%)")
    axis.grid(True, alpha=0.25)
    _save(figure, outputs["drawdown"])

    figure, axis = plt.subplots(figsize=(11, 4))
    if not annual_returns.empty:
        years = annual_returns["period"].astype(str).tolist()
        values = annual_returns["return"].to_numpy() * 100
        axis.bar(years, values, color=np.where(values >= 0, "#2a9d8f", "#e76f51"))
        axis.tick_params(axis="x", rotation=60)
    axis.set(title="Rendements par année civile", xlabel="Année", ylabel="Rendement (%)")
    axis.axhline(0, color="black", linewidth=0.7)
    axis.grid(True, axis="y", alpha=0.25)
    _save(figure, outputs["annual"])

    matrix = monthly_return_matrix(monthly_returns)
    figure, axis = plt.subplots(figsize=(11, max(4, len(matrix) * 0.35)))
    if not matrix.empty:
        values = matrix.to_numpy(dtype=float) * 100
        image = axis.imshow(values, aspect="auto", cmap="RdYlGn", interpolation="nearest")
        axis.set_yticks(range(len(matrix.index)), labels=[str(year) for year in matrix.index])
        axis.set_xticks(range(12), labels=["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"])
        figure.colorbar(image, ax=axis, label="Rendement (%)")
    axis.set(title="Rendements mensuels")
    _save(figure, outputs["monthly"])

    figure, axis = plt.subplots(figsize=(11, 5))
    if not allocations.empty:
        weights = allocations.pivot(index="market_date", columns="currency", values="weight")
        weights.index = pd.to_datetime(weights.index)
        weights = weights.sort_index()
        if set(weights.index) != set(nav.index):
            raise ValueError("Allocation and NAV histories must cover the same analysis dates")
        if weights.isna().any().any() or not np.isfinite(weights.to_numpy()).all():
            raise ValueError("Allocation history contains missing or non-finite weights")
        if not np.allclose(weights.sum(axis=1).to_numpy(), 1.0, rtol=0.0, atol=1e-8):
            raise ValueError("Allocation weights must sum to one at every market date")
        axis.stackplot(
            weights.index,
            [weights[column].to_numpy() for column in weights.columns],
            labels=weights.columns,
            alpha=0.85,
        )
        axis.legend(loc="upper left", ncol=4, fontsize="small")
    axis.set(title="Poids réalisés des devises", xlabel="Date", ylabel="Poids")
    axis.set_ylim(0, 1)
    axis.grid(True, alpha=0.2)
    _save(figure, outputs["weights"])

    figure, axis = plt.subplots(figsize=(11, 4))
    if not transactions.empty:
        fees = transactions.copy()
        fees["execution_date"] = pd.to_datetime(fees["execution_date"])
        daily_fees = fees.groupby("execution_date")["fee_eur"].sum()
        cumulative = daily_fees.reindex(nav.index).fillna(0.0).cumsum()
    else:
        cumulative = pd.Series(0.0, index=nav.index)
    if not cumulative.empty:
        axis.step(cumulative.index, cumulative.to_numpy(), where="post", linewidth=1.1)
    axis.set(title="Frais cumulés", xlabel="Date d'exécution", ylabel="Frais (EUR)")
    axis.grid(True, alpha=0.25)
    _save(figure, outputs["fees"])
    return outputs


def _save(figure: Figure, path: Path) -> None:
    figure.tight_layout()
    figure.savefig(path, dpi=140)
    plt.close(figure)
