"""Run and report the step-7B reference-strategy comparison."""

from __future__ import annotations

import argparse
import json
import math
from collections.abc import Mapping, Sequence
from datetime import date, timedelta
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import pandas as pd
import yaml

from forex_v1.backtest.config import BacktestConfig
from forex_v1.backtest.data import MarketData, load_market_data
from forex_v1.backtest.engine import BacktestResult, build_history_frame, run_backtest
from forex_v1.data.layout import PROJECT_ROOT
from forex_v1.data.schemas import SUPPORTED_CURRENCIES
from forex_v1.evaluation.metrics import calculate_metrics
from forex_v1.evaluation.risk import drawdown_series
from forex_v1.strategies import (
    BENCHMARK_NAMES,
    BENCHMARK_REBALANCE_POLICIES,
    create_benchmark_provider,
    required_features_for_benchmark,
)

METRICS_TO_COMPARE = (
    "cagr",
    "total_return",
    "maximum_drawdown",
    "sharpe_ratio",
    "sortino_ratio",
    "calmar_ratio",
    "annualized_volatility",
    "total_fees_eur",
    "turnover",
    "final_nav_eur",
)
REQUIRED_FOREIGN_CURRENCIES = tuple(sorted(SUPPORTED_CURRENCIES - {"EUR"}))


def _read_comparison_config(path: str | Path) -> dict[str, Any]:
    try:
        document = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    except OSError as error:
        raise ValueError(f"Cannot read benchmark config {path}: {error}") from error
    except yaml.YAMLError as error:
        raise ValueError(f"Invalid benchmark YAML: {error}") from error
    if not isinstance(document, Mapping):
        raise TypeError("Benchmark config root must be a mapping")
    portfolio = document.get("portfolio")
    scenario = document.get("scenario")
    if not isinstance(portfolio, Mapping) or not isinstance(scenario, Mapping):
        raise TypeError("Benchmark config must contain portfolio and scenario mappings")
    capital = portfolio.get("initial_capital_eur")
    transaction_cost = scenario.get("transaction_cost_bps")
    availability_lag = scenario.get("availability_lag_days")
    development_fraction = scenario.get("development_fraction")
    max_foreign_weight = scenario.get("max_foreign_weight")
    strategies = scenario.get("strategies")
    if isinstance(capital, bool) or not isinstance(capital, (int, float)) or capital <= 0:
        raise ValueError("portfolio.initial_capital_eur must be positive and numeric")
    if (
        isinstance(transaction_cost, bool)
        or not isinstance(transaction_cost, (int, float))
        or not math.isfinite(float(transaction_cost))
        or transaction_cost < 0
    ):
        raise ValueError("scenario.transaction_cost_bps must be finite and non-negative")
    if (
        isinstance(availability_lag, bool)
        or not isinstance(availability_lag, int)
        or availability_lag < 7
    ):
        raise ValueError("scenario.availability_lag_days must be an integer of at least 7")
    if (
        isinstance(development_fraction, bool)
        or not isinstance(development_fraction, (int, float))
        or not 0 < development_fraction < 1
    ):
        raise ValueError("scenario.development_fraction must be strictly between zero and one")
    if (
        isinstance(max_foreign_weight, bool)
        or not isinstance(max_foreign_weight, (int, float))
        or not math.isfinite(float(max_foreign_weight))
        or not 0 < max_foreign_weight <= 1
    ):
        raise ValueError("scenario.max_foreign_weight must be finite and in (0, 1]")
    if (
        not isinstance(strategies, list)
        or not strategies
        or any(not isinstance(name, str) or name not in BENCHMARK_NAMES for name in strategies)
        or len(strategies) != len(set(strategies))
    ):
        raise ValueError("scenario.strategies must be a unique list of supported benchmark names")
    return {
        "initial_capital_eur": float(capital),
        "transaction_cost_bps": float(transaction_cost),
        "availability_lag_days": availability_lag,
        "development_fraction": float(development_fraction),
        "max_foreign_weight": float(max_foreign_weight),
        "strategies": tuple(strategies),
    }


def _complete_feature_row(
    market_data: MarketData,
    market_date: date,
    feature_names: Sequence[str],
) -> bool:
    values = market_data.features_by_date.get(market_date, {})
    return all(
        (value := values.get(f"{currency}__{feature_name}")) is not None
        and math.isfinite(float(value))
        for currency in REQUIRED_FOREIGN_CURRENCIES
        for feature_name in feature_names
    )


def _common_development_period(
    market_data: MarketData,
    *,
    development_fraction: float,
    availability_lag_days: int,
    strategy_names: Sequence[str],
) -> tuple[date, date, date, date, int]:
    split_index = int(len(market_data.dates) * development_fraction)
    if split_index < 2:
        raise ValueError("Not enough common market observations for a development period")
    development_dates = market_data.dates[:split_index]
    development_end = development_dates[-1]
    required_features = tuple(
        sorted(
            {
                feature
                for strategy_name in strategy_names
                for feature in required_features_for_benchmark(strategy_name)
            }
        )
    )
    if not required_features:
        first_complete_feature_date = development_dates[0]
    else:
        unavailable = set(required_features) - set(market_data.feature_names)
        if unavailable:
            raise ValueError(f"Feature history is missing required columns: {sorted(unavailable)}")
        first_complete_feature_date = next(
            (
                market_date
                for market_date in development_dates
                if _complete_feature_row(market_data, market_date, required_features)
            ),
            None,
        )
        if first_complete_feature_date is None:
            raise ValueError(
                "No common development date has admissible values for all required features"
            )

    earliest_signal = first_complete_feature_date + timedelta(days=availability_lag_days)
    first_signal = earliest_signal + timedelta(days=(4 - earliest_signal.weekday()) % 7)
    while first_signal <= development_end:
        cutoff = first_signal - timedelta(days=availability_lag_days)
        eligible_dates = [market_date for market_date in development_dates if market_date <= cutoff]
        if eligible_dates and _complete_feature_row(
            market_data,
            eligible_dates[-1],
            required_features,
        ):
            break
        first_signal += timedelta(days=7)
    if first_signal > development_end:
        raise ValueError("No eligible common Friday signal exists in the development period")
    information_date = max(
        market_date for market_date in development_dates
        if market_date <= first_signal - timedelta(days=availability_lag_days)
    )
    first_execution = next(
        (market_date for market_date in development_dates if market_date > first_signal),
        None,
    )
    if first_execution is None:
        raise ValueError("The development period ends before a common signal can execute")
    return (
        first_signal,
        information_date,
        first_execution,
        development_end,
        split_index,
    )


def _transaction_frame(result: BacktestResult) -> pd.DataFrame:
    rows = [
        {
            "signal_date": execution.signal_date.isoformat(),
            "execution_date": execution.execution_date.isoformat(),
            "side": transaction.side,
            "currency": transaction.currency,
            "quantity": transaction.quantity,
            "notional_eur": transaction.notional_eur,
            "fee_eur": transaction.fee_eur,
        }
        for execution in result.executions
        for transaction in execution.rebalance.transactions
    ]
    return pd.DataFrame(
        rows,
        columns=[
            "signal_date",
            "execution_date",
            "side",
            "currency",
            "quantity",
            "notional_eur",
            "fee_eur",
        ],
    )


def _allocation_frame(result: BacktestResult, market_data: MarketData) -> pd.DataFrame:
    rows = []
    for record in result.daily_records:
        prices = market_data.prices_by_date[record.market_date]
        for currency in sorted(SUPPORTED_CURRENCIES):
            rows.append(
                {
                    "market_date": record.market_date.isoformat(),
                    "currency": currency,
                    "quantity": record.cash_eur if currency == "EUR" else record.units[currency],
                    "eur_per_unit": prices[currency],
                    "value_eur": record.values_eur[currency],
                    "weight": record.weights[currency],
                }
            )
    return pd.DataFrame(
        rows,
        columns=["market_date", "currency", "quantity", "eur_per_unit", "value_eur", "weight"],
    )


def _write_strategy_outputs(
    result: BacktestResult,
    market_data: MarketData,
    output_dir: Path,
) -> dict[str, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    nav = pd.DataFrame(
        [
            {
                "market_date": record.market_date.isoformat(),
                "nav_eur": record.nav_eur,
                "cash_eur": record.cash_eur,
            }
            for record in result.daily_records
        ]
    )
    allocations = _allocation_frame(result, market_data)
    decisions = pd.DataFrame(
        [
            {
                "signal_date": item.signal_date.isoformat(),
                "information_cutoff_date": item.information_cutoff_date.isoformat(),
                "information_market_date": (
                    item.information_market_date.isoformat()
                    if item.information_market_date
                    else ""
                ),
                "execution_date": item.execution_date.isoformat() if item.execution_date else "",
                "status": item.status,
                "target_weights": (
                    json.dumps(item.target_weights, sort_keys=True, separators=(",", ":"))
                    if item.target_weights is not None
                    else ""
                ),
            }
            for item in result.decisions
        ]
    )
    transactions = _transaction_frame(result)
    outputs = {
        "nav": output_dir / "nav.csv",
        "allocations": output_dir / "allocations.csv",
        "decisions": output_dir / "decisions.csv",
        "transactions": output_dir / "transactions.csv",
    }
    nav.to_csv(outputs["nav"], index=False, float_format="%.12g")
    allocations.to_csv(outputs["allocations"], index=False, float_format="%.12g")
    decisions.to_csv(outputs["decisions"], index=False)
    transactions.to_csv(outputs["transactions"], index=False, float_format="%.12g")
    return outputs


def _metric_rows(
    result: BacktestResult,
    transactions: pd.DataFrame,
) -> dict[str, Any]:
    metrics = calculate_metrics(result.nav_series, transactions=transactions)
    metrics["final_nav_eur"] = result.final_nav_eur
    return metrics


def _write_charts(
    results: Mapping[str, BacktestResult],
    allocations: Mapping[str, pd.DataFrame],
    metrics: Mapping[str, Mapping[str, Any]],
    output_dir: Path,
) -> dict[str, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    capital_path = output_dir / "benchmark_capital_curves.png"
    drawdown_path = output_dir / "benchmark_drawdowns.png"
    table_path = output_dir / "benchmark_metrics_table.png"
    allocation_path = output_dir / "benchmark_dynamic_allocations.png"

    figure, axis = plt.subplots(figsize=(12, 6))
    for name, result in results.items():
        axis.plot(pd.to_datetime(result.nav_series.index), result.nav_series, label=name)
    axis.set(title="Capital EUR — stratégies de référence (développement uniquement)", xlabel="Date", ylabel="NAV (EUR)")
    axis.grid(True, alpha=0.25)
    axis.legend(loc="best", fontsize="small")
    figure.tight_layout()
    figure.savefig(capital_path, dpi=140)
    plt.close(figure)

    figure, axis = plt.subplots(figsize=(12, 5))
    for name, result in results.items():
        values = drawdown_series(result.nav_series)
        axis.plot(pd.to_datetime(values.index), values * 100, label=name)
    axis.set(title="Drawdowns comparés — développement uniquement", xlabel="Date", ylabel="Drawdown (%)")
    axis.grid(True, alpha=0.25)
    axis.legend(loc="best", fontsize="small")
    figure.tight_layout()
    figure.savefig(drawdown_path, dpi=140)
    plt.close(figure)

    ordered_names = sorted(
        metrics,
        key=lambda name: (
            metrics[name]["cagr"] is None,
            -(metrics[name]["cagr"] if metrics[name]["cagr"] is not None else 0.0),
        ),
    )
    headers = ["Stratégie", "CAGR", "Total", "MDD", "Sharpe", "Sortino", "Frais EUR"]
    cells = [
        [
            name,
            _percent(metrics[name]["cagr"]),
            _percent(metrics[name]["total_return"]),
            _percent(metrics[name]["maximum_drawdown"]),
            _number(metrics[name]["sharpe_ratio"]),
            _number(metrics[name]["sortino_ratio"]),
            _number(metrics[name]["total_fees_eur"]),
        ]
        for name in ordered_names
    ]
    figure, axis = plt.subplots(figsize=(13, max(3.5, 0.55 * (len(cells) + 2))))
    axis.axis("off")
    axis.table(cellText=cells, colLabels=headers, loc="center", cellLoc="center")
    axis.set_title("Classement descriptif par CAGR — période de développement", pad=18)
    figure.tight_layout()
    figure.savefig(table_path, dpi=140, bbox_inches="tight")
    plt.close(figure)

    chart_paths = {
        "capital": capital_path,
        "drawdowns": drawdown_path,
        "metrics_table": table_path,
    }
    dynamic_names = [
        name for name in ("momentum_60", "inverse_volatility_60") if name in allocations
    ]
    if dynamic_names:
        figure, axes = plt.subplots(
            len(dynamic_names),
            1,
            figsize=(12, max(4, 4.5 * len(dynamic_names))),
            sharex=True,
            squeeze=False,
        )
        flat_axes = axes[:, 0]
        for axis, name in zip(flat_axes, dynamic_names, strict=True):
            frame = allocations[name].copy()
            pivot = frame.pivot(index="market_date", columns="currency", values="weight")
            pivot.index = pd.to_datetime(pivot.index)
            axis.stackplot(
                pivot.index,
                [pivot[currency].to_numpy() for currency in sorted(pivot.columns)],
                labels=sorted(pivot.columns),
                alpha=0.85,
            )
            axis.set(title=name.replace("_", " ").title(), ylabel="Poids réalisé")
            axis.set_ylim(0, 1)
            axis.grid(True, alpha=0.2)
        flat_axes[-1].set_xlabel("Date")
        flat_axes[0].legend(loc="upper left", ncol=4, fontsize="small")
        figure.suptitle("Allocations réalisées des stratégies dynamiques")
        figure.tight_layout()
        figure.savefig(allocation_path, dpi=140)
        plt.close(figure)
        chart_paths["dynamic_allocations"] = allocation_path
    return chart_paths


def _number(value: Any, decimals: int = 4) -> str:
    return "N/D" if value is None else f"{float(value):,.{decimals}f}"


def _percent(value: Any) -> str:
    return "N/D" if value is None else f"{float(value):.2%}"


def _render_markdown(payload: Mapping[str, Any]) -> str:
    period = payload["period"]
    rows = payload["strategies"]
    ordered = sorted(
        (row for row in rows.values() if row["status"] == "completed"),
        key=lambda row: (
            row["metrics"]["cagr"] is None,
            -(row["metrics"]["cagr"] if row["metrics"]["cagr"] is not None else 0.0),
        ),
    )
    table = [
        "| Rang | Stratégie | CAGR | Rendement total | MDD | >20 % | Sharpe | Sortino | Calmar | Vol. ann. | Frais EUR | Turnover | NAV finale EUR |",
        "|---:|---|---:|---:|---:|:---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for rank, row in enumerate(ordered, start=1):
        metrics = row["metrics"]
        table.append(
            f"| {rank} | {row['display_name']} | {_percent(metrics['cagr'])} | "
            f"{_percent(metrics['total_return'])} | {_percent(metrics['maximum_drawdown'])} | "
            f"{'Oui' if metrics['maximum_drawdown'] is not None and abs(metrics['maximum_drawdown']) > 0.20 else 'Non'} | "
            f"{_number(metrics['sharpe_ratio'])} | {_number(metrics['sortino_ratio'])} | "
            f"{_number(metrics['calmar_ratio'])} | {_percent(metrics['annualized_volatility'])} | "
            f"{_number(metrics['total_fees_eur'], 2)} | {_number(metrics['turnover'])} | "
            f"{_number(metrics['final_nav_eur'], 2)} |"
        )
    strategy_details = []
    for row in rows.values():
        if row["status"] != "completed":
            strategy_details.append(
                f"- **{row['display_name']}** : échec conservé — `{row['error']}`."
            )
            continue
        strategy_details.append(
            f"- **{row['display_name']}** : {row['observation_count']:,} observations, "
            f"{row['decision_count']:,} décisions (y compris signaux ignorés), "
            f"{row['rebalance_count']:,} rééquilibrages avec transactions, "
            f"{row['transaction_count']:,} transactions ; première exécution "
            f"{row['first_execution_date'] or 'aucune'}."
        )
    charts = payload["charts"]
    return f"""# Comparaison des stratégies de référence (étape 7B)

## Périmètre et dates

- Développement chronologique : **{period['start_date']} à {period['end_date']}** inclus
  ({period['development_observation_count']:,} observations disponibles avant warm-up).
- Première feature commune admissible : {period['first_complete_feature_date']}.
- Premier signal commun : **{period['first_signal_date']}**, fondé sur les observations
  jusqu'au **{period['first_information_date']}** (date de marché au plus tard au cutoff
  du signal moins {period['availability_lag_days']} jours calendaires).
- Première exécution possible : **{period['first_execution_date']}**, première observation
  strictement postérieure au signal.
- Période NAV commune après warm-up : **{period['nav_start_date']} à {period['nav_end_date']}**.
- Toutes les stratégies démarrent avec {payload['configuration']['initial_capital_eur']:,.2f} EUR,
  le même univers, les mêmes observations et les mêmes dates de signal. Frais :
  {payload['configuration']['transaction_cost_bps']:g} bps par notionnel converti.
- Seules les métriques jusqu'à la frontière développement du **{period['end_date']}** sont
  calculées ou classées. La période de test final de 25 % est exclue de ce rapport.

## Classement descriptif par CAGR

{chr(10).join(table)}

Le tableau est un diagnostic sur une seule période de développement, et ne désigne pas
une stratégie gagnante. L'exposition antérieure au backtest fixe 6B sur toute la période
historique est documentée ; cette analyse ne prétend donc pas que le test final a été
entièrement aveugle à toute analyse antérieure.

## Décisions et transactions

{chr(10).join(strategy_details)}

Les signaux restent datés le vendredi calendaire ; un vendredi férié ou sans observation
ne supprime pas la décision, et l'exécution a lieu à la première observation ultérieure.
Buy & Hold n'est exécutée qu'une fois ; ses poids dérivent ensuite avec le marché.

## Graphiques

{chr(10).join(f"- `{path}`" for path in charts.values())}

## Hypothèses et limites

- Les dates de marché ne sont pas des dates de publication : les dates réelles de
  disponibilité des séries FRED sont inconnues. Le délai de 7 jours est une hypothèse
  conservatrice de recherche et non une garantie point-in-time.
- Momentum 60 utilise les log-rendements historiques à 60 observations : trois devises
  étrangères au plus, strictement positives, à 25 % chacune ; les emplacements inutilisés
  restent en EUR.
- Inverse Volatility 60 exclut volatilités absentes, non finies, nulles ou négatives ; le
  plafond cible est de {payload['configuration']['max_foreign_weight']:.0%} par devise et le
  surplus reste en EUR. Les poids réalisés peuvent dériver au-dessus du plafond entre
  deux rééquilibrages hebdomadaires.
- Les taux FRED sont des références indicatives, pas des prix exécutables certifiés.
  Aucun levier, short, intérêt, carry, spread ou glissement n'est simulé.
- Ce classement ne doit pas servir à optimiser des modèles sur la période finale.
"""


def run_benchmark_comparison(
    market_data: MarketData,
    *,
    configuration: Mapping[str, Any],
    output_dir: str | Path,
) -> dict[str, Any]:
    """Run all configured reference strategies on a shared development window."""

    names = configuration["strategies"]
    first_signal, information_date, first_execution, development_end, split_index = (
        _common_development_period(
            market_data,
            development_fraction=configuration["development_fraction"],
            availability_lag_days=configuration["availability_lag_days"],
            strategy_names=names,
        )
    )
    development_dates = market_data.dates[:split_index]
    common_config = BacktestConfig(
        initial_capital_eur=configuration["initial_capital_eur"],
        transaction_cost_bps=configuration["transaction_cost_bps"],
        availability_lag_days=configuration["availability_lag_days"],
        target_weights={"EUR": 1.0},
        start_date=first_signal,
        end_date=development_end,
    )
    output_root = Path(output_dir)
    strategy_root = output_root / "benchmarks"
    strategy_root.mkdir(parents=True, exist_ok=True)
    shared_history = build_history_frame(market_data, development_end)
    results: dict[str, BacktestResult] = {}
    allocations: dict[str, pd.DataFrame] = {}
    reports: dict[str, dict[str, Any]] = {}

    for name in names:
        policy = BENCHMARK_REBALANCE_POLICIES[name]
        config = BacktestConfig(
            initial_capital_eur=common_config.initial_capital_eur,
            transaction_cost_bps=common_config.transaction_cost_bps,
            availability_lag_days=common_config.availability_lag_days,
            target_weights=common_config.target_weights,
            start_date=common_config.start_date,
            end_date=common_config.end_date,
            strategy_name=name,
            rebalance_policy=policy,
        )
        provider = create_benchmark_provider(
            name,
            max_foreign_weight=configuration["max_foreign_weight"],
        )
        try:
            result = run_backtest(
                market_data,
                provider,
                config,
                history_frame=shared_history,
            )
            for record in result.daily_records:
                if any(weight < 0 or weight > 1 for weight in record.weights.values()):
                    raise ArithmeticError(f"{name} generated an invalid realized allocation")
                if not math.isclose(
                    math.fsum(record.weights.values()),
                    1.0,
                    rel_tol=0.0,
                    abs_tol=1e-8,
                ):
                    raise ArithmeticError(f"{name} realized weights do not sum to one")
            if name == "inverse_volatility_60":
                for decision in result.decisions:
                    if decision.target_weights is None:
                        continue
                    if any(
                        decision.target_weights.get(currency, 0.0)
                        > configuration["max_foreign_weight"] + 1e-9
                        for currency in REQUIRED_FOREIGN_CURRENCIES
                    ):
                        raise ArithmeticError(
                            "Inverse Volatility 60 target exceeds the configured foreign cap"
                        )
            transaction_frame = _transaction_frame(result)
            metrics = _metric_rows(result, transaction_frame)
            if not math.isclose(
                metrics["total_fees_eur"],
                result.total_fees_eur,
                rel_tol=0.0,
                abs_tol=1e-8,
            ):
                raise ArithmeticError(f"{name} transaction fees do not reconcile")
            allocations_frame = _allocation_frame(result, market_data)
            paths = _write_strategy_outputs(
                result,
                market_data,
                strategy_root / name,
            )
            results[name] = result
            allocations[name] = allocations_frame
            reports[name] = {
                "status": "completed",
                "display_name": name.replace("_", " ").title(),
                "rebalance_policy": policy,
                "observation_count": len(result.daily_records),
                "decision_count": len(result.decisions),
                "execution_count": len(result.executions),
                "rebalance_count": sum(
                    bool(execution.rebalance.transactions) for execution in result.executions
                ),
                "transaction_count": len(transaction_frame),
                "first_execution_date": (
                    result.executions[0].execution_date.isoformat()
                    if result.executions
                    else None
                ),
                "total_fees_eur": result.total_fees_eur,
                "metrics": metrics,
                "files": {key: str(path.relative_to(output_root)) for key, path in paths.items()},
            }
        except Exception as error:  # noqa: BLE001 - preserve per-strategy failures in the report
            reports[name] = {
                "status": "failed",
                "display_name": name.replace("_", " ").title(),
                "error": f"{type(error).__name__}: {error}",
            }

    if not results:
        raise RuntimeError("Every configured benchmark failed; see strategy failure records")
    expected_nav_dates = tuple(next(iter(results.values())).nav_series.index)
    expected_signal_dates = tuple(
        decision.signal_date for decision in next(iter(results.values())).decisions
    )
    for name, result in results.items():
        if tuple(result.nav_series.index) != expected_nav_dates:
            raise RuntimeError(f"{name} did not produce the common NAV calendar")
        if tuple(decision.signal_date for decision in result.decisions) != expected_signal_dates:
            raise RuntimeError(f"{name} did not produce the common signal calendar")
    weekly_results = {
        name: result
        for name, result in results.items()
        if result.config.rebalance_policy == "weekly"
    }
    expected_weekly_execution_dates = tuple(
        execution.execution_date for execution in next(iter(weekly_results.values())).executions
    )
    for name, result in weekly_results.items():
        if tuple(execution.execution_date for execution in result.executions) != (
            expected_weekly_execution_dates
        ):
            raise RuntimeError(f"{name} did not produce the common weekly execution calendar")
    first_common_execution = expected_weekly_execution_dates[0]
    for name, result in results.items():
        if not result.executions or result.executions[0].execution_date != first_common_execution:
            raise RuntimeError(f"{name} did not share the common first execution date")
    charts = _write_charts(
        results,
        allocations,
        {
            name: row["metrics"]
            for name, row in reports.items()
            if row["status"] == "completed"
        },
        output_root,
    )

    first_record = next(iter(results.values())).daily_records[0]
    last_record = next(iter(results.values())).daily_records[-1]
    source_hashes = {
        "eur": market_data.eur_source_sha256,
        "features": market_data.feature_source_sha256,
    }
    payload: dict[str, Any] = {
        "configuration": {
            key: list(value) if isinstance(value, tuple) else value
            for key, value in configuration.items()
        },
        "period": {
            "start_date": development_dates[0].isoformat(),
            "end_date": development_end.isoformat(),
            "development_observation_count": split_index,
            "first_complete_feature_date": min(
                market_date
                for market_date in development_dates
                if _complete_feature_row(
                    market_data,
                    market_date,
                    tuple(sorted({
                        feature
                        for name in names
                        for feature in required_features_for_benchmark(name)
                    })),
                )
            ).isoformat(),
            "first_signal_date": first_signal.isoformat(),
            "first_information_date": information_date.isoformat(),
            "first_execution_date": first_execution.isoformat(),
            "nav_start_date": first_record.market_date.isoformat(),
            "nav_end_date": last_record.market_date.isoformat(),
            "availability_lag_days": configuration["availability_lag_days"],
            "final_test_period_evaluated": False,
        },
        "source_sha256": source_hashes,
        "strategies": reports,
        "charts": {name: str(path.relative_to(output_root)) for name, path in charts.items()},
        "comparison_validation": {
            "common_nav_dates": True,
            "common_signal_dates": True,
            "common_weekly_execution_dates": True,
            "common_first_execution_date": first_common_execution.isoformat(),
            "comparison_date_count": len(expected_nav_dates),
            "comparison_signal_count": len(expected_signal_dates),
        },
    }
    json_path = output_root / "benchmark_comparison.json"
    markdown_path = output_root / "benchmark_comparison.md"
    json_path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    markdown_path.write_text(_render_markdown(payload), encoding="utf-8")
    payload["report_path"] = str(markdown_path)
    payload["json_path"] = str(json_path)
    return payload


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--eur-parquet",
        type=Path,
        default=PROJECT_ROOT / "data" / "derived" / "eur_rates",
    )
    parser.add_argument(
        "--features-parquet",
        type=Path,
        default=PROJECT_ROOT / "data" / "derived" / "features",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=PROJECT_ROOT / "configs" / "benchmarks.yaml",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "reports",
    )
    arguments = parser.parse_args(argv)
    configuration = _read_comparison_config(arguments.config)
    market_data = load_market_data(arguments.eur_parquet, arguments.features_parquet)
    result = run_benchmark_comparison(
        market_data,
        configuration=configuration,
        output_dir=arguments.output_dir,
    )
    print(
        f"Benchmark report: {result['report_path']}; "
        f"development {result['period']['start_date']}..{result['period']['end_date']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
