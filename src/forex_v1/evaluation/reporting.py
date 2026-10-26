"""Load existing step-6B exports and produce reusable performance reports."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from datetime import date
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml

from forex_v1.data.layout import PROJECT_ROOT
from forex_v1.evaluation.metrics import analyze_performance, calculate_metrics, validate_nav
from forex_v1.evaluation.plots import create_performance_plots
from forex_v1.evaluation.risk import drawdown_episodes

PREDEFINED_PERIODS = (
    ("2000-2009", date(2000, 1, 1), date(2009, 12, 31)),
    ("2010-2019", date(2010, 1, 1), date(2019, 12, 31)),
    ("2020-2026", date(2020, 1, 1), date(2026, 12, 31)),
)
METRIC_LABELS = {
    "total_return": "Total Return",
    "cagr": "CAGR",
    "maximum_drawdown": "Maximum Drawdown",
    "sharpe_ratio": "Sharpe Ratio",
    "sortino_ratio": "Sortino Ratio",
    "calmar_ratio": "Calmar Ratio",
    "annualized_volatility": "Annualized Volatility",
    "best_year_return": "Best Year",
    "worst_year_return": "Worst Year",
    "best_month_return": "Best Month",
    "worst_month_return": "Worst Month",
    "win_rate": "Win Rate",
    "turnover": "Turnover",
    "total_fees_eur": "Total Fees (EUR)",
}


def load_backtest_exports(
    nav_path: str | Path,
    transactions_path: str | Path,
    allocations_path: str | Path,
) -> tuple[pd.Series, pd.DataFrame, pd.DataFrame]:
    """Load only the existing CSV result exports; does not run the backtester."""

    nav_frame = pd.read_csv(nav_path, parse_dates=["market_date"])
    transaction_frame = pd.read_csv(transactions_path)
    allocation_frame = pd.read_csv(allocations_path, parse_dates=["market_date"])
    if not {"market_date", "nav_eur"}.issubset(nav_frame.columns):
        raise ValueError("NAV export must contain market_date and nav_eur")
    if not {"market_date", "currency", "weight"}.issubset(allocation_frame.columns):
        raise ValueError("Allocation export must contain market_date, currency, and weight")
    nav = nav_frame.set_index("market_date")["nav_eur"].rename("nav_eur")
    return nav, transaction_frame, allocation_frame


def run_performance_analysis(
    *,
    nav_path: str | Path,
    transactions_path: str | Path,
    allocations_path: str | Path,
    backtest_report_path: str | Path,
    config_path: str | Path,
    output_dir: str | Path,
    start_date: str | None = None,
    end_date: str | None = None,
    periods_per_year: int = 252,
) -> dict[str, Path]:
    """Analyze existing backtest exports and write Markdown, JSON, and plots."""

    nav_raw, transactions_raw, allocations_raw = load_backtest_exports(
        nav_path,
        transactions_path,
        allocations_path,
    )
    nav, missing_count = validate_nav(
        nav_raw,
        start_date=start_date,
        end_date=end_date,
    )
    nav_dates = pd.DatetimeIndex(nav.index)
    analysis = analyze_performance(
        nav,
        transactions=transactions_raw,
        periods_per_year=periods_per_year,
    )
    metrics = analysis["metrics"]
    metrics["missing_nav_observations_dropped"] = missing_count
    transactions = _filter_transactions(transactions_raw, nav.index[0], nav.index[-1])
    allocations = allocations_raw.loc[
        (allocations_raw["market_date"] >= nav_dates[0])
        & (allocations_raw["market_date"] <= nav_dates[-1])
    ].copy()

    config_document = _read_yaml_mapping(config_path)
    source_paths = {
        "nav": Path(nav_path),
        "transactions": Path(transactions_path),
        "allocations": Path(allocations_path),
        "backtest_report": Path(backtest_report_path),
        "config": Path(config_path),
    }
    source_hashes = {
        name: hashlib.sha256(path.read_bytes()).hexdigest()
        for name, path in source_paths.items()
    }

    period_metrics: dict[str, Any] = {}
    for label, start, end in PREDEFINED_PERIODS:
        selected = nav.loc[(nav_dates.date >= start) & (nav_dates.date <= end)]
        if selected.empty:
            period_metrics[label] = None
            continue
        period_metrics[label] = calculate_metrics(
            selected,
            transactions=transactions_raw,
            periods_per_year=periods_per_year,
        )
    if start_date is not None or end_date is not None:
        selected_label = (
            f"selected:{start_date or nav_dates[0].date()}.."
            f"{end_date or nav_dates[-1].date()}"
        )
        period_metrics[selected_label] = calculate_metrics(
            nav_raw,
            transactions=transactions_raw,
            periods_per_year=periods_per_year,
            start_date=start_date,
            end_date=end_date,
        )

    annual_returns = analysis["annual_returns"]
    monthly_returns = analysis["monthly_returns"]
    drawdowns = drawdown_episodes(nav)
    output_root = Path(output_dir)
    output_root.mkdir(parents=True, exist_ok=True)
    plots = create_performance_plots(
        nav,
        annual_returns,
        monthly_returns,
        allocations,
        transactions,
        output_root,
    )
    json_path = output_root / "performance_metrics.json"
    report_path = output_root / "performance_analysis.md"
    payload = {
        "configuration": _json_safe(config_document),
        "source_hashes_sha256": source_hashes,
        "analysis": {
            "start_date": nav_dates[0].date().isoformat(),
            "end_date": nav_dates[-1].date().isoformat(),
            "periods_per_year": periods_per_year,
            "metrics": metrics,
            "annual_returns": _records(annual_returns),
            "monthly_returns": _records(monthly_returns),
            "largest_drawdowns": _records(drawdowns),
            "predefined_periods": period_metrics,
            "temporal_split_note": (
                "Analyse descriptive uniquement. La frontière chronologique "
                "75 % développement / 25 % test final doit rester intacte pour "
                "la sélection future des modèles."
            ),
        },
        "charts": {key: path.name for key, path in plots.items()},
    }
    json_path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    report_path.write_text(
        _render_report(payload, annual_returns, drawdowns),
        encoding="utf-8",
    )
    return {"report": report_path, "json": json_path, **plots}


def _filter_transactions(
    transactions: pd.DataFrame,
    start: pd.Timestamp,
    end: pd.Timestamp,
) -> pd.DataFrame:
    if transactions.empty:
        return transactions.copy()
    required = {"execution_date", "notional_eur", "fee_eur"}
    missing = required - set(transactions.columns)
    if missing:
        raise ValueError(f"Transaction export missing columns: {sorted(missing)}")
    dates = pd.to_datetime(transactions["execution_date"], errors="raise")
    if start.tz is not None and dates.dt.tz is None:
        dates = dates.dt.tz_localize(start.tz)
    elif start.tz is None and dates.dt.tz is not None:
        dates = dates.dt.tz_localize(None)
    return transactions.loc[(dates >= start) & (dates <= end)].copy()


def _read_yaml_mapping(path: str | Path) -> dict[str, Any]:
    try:
        value = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError) as error:
        raise ValueError(f"Cannot read evaluation configuration {path}: {error}") from error
    if not isinstance(value, dict):
        raise TypeError("Backtest configuration root must be a mapping")
    return value


def _records(frame: pd.DataFrame) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for record in frame.to_dict(orient="records"):
        normalized: dict[str, Any] = {}
        for key, value in record.items():
            field_name = str(key)
            if pd.isna(value):
                normalized[field_name] = None
            elif isinstance(value, (pd.Timestamp, date)):
                normalized[field_name] = value.isoformat()
            elif isinstance(value, (np.integer,)):
                normalized[field_name] = int(value)
            elif isinstance(value, (np.floating,)):
                normalized[field_name] = float(value)
            else:
                normalized[field_name] = value
        records.append(normalized)
    return records


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_json_safe(item) for item in value]
    if isinstance(value, (date, pd.Timestamp)):
        return value.isoformat()
    return value


def _format_metric(value: Any, key: str) -> str:
    if value is None or (isinstance(value, float) and not math.isfinite(value)):
        return "N/D"
    if key in {
        "total_return",
        "cagr",
        "maximum_drawdown",
        "annualized_volatility",
        "best_year_return",
        "worst_year_return",
        "best_month_return",
        "worst_month_return",
        "win_rate",
    }:
        return f"{value:.2%}"
    if key == "total_fees_eur":
        return f"{value:,.2f} EUR"
    if key == "turnover":
        return f"{value:.3f}x"
    return f"{value:.4f}"


def _render_report(
    payload: dict[str, Any],
    annual_returns: pd.DataFrame,
    drawdowns: pd.DataFrame,
) -> str:
    analysis = payload["analysis"]
    metrics = analysis["metrics"]
    lines = [
        "# Analyse des performances du backtest",
        "",
        "## Configuration",
        "",
        "```yaml",
        yaml.safe_dump(payload["configuration"], sort_keys=True, allow_unicode=True).rstrip(),
        "```",
        "",
        "## Période et capital",
        "",
        f"- Période analysée : {analysis['start_date']} – {analysis['end_date']}.",
        f"- Observations NAV valides : {metrics['observation_count']:,}.",
        f"- Capital initial : {metrics['initial_nav_eur']:,.2f} EUR.",
        f"- Capital final : {metrics['final_nav_eur']:,.2f} EUR.",
        f"- Observations NAV manquantes exclues : {metrics['missing_nav_observations_dropped']}.",
        "",
        "## Métriques",
        "",
        "| Métrique | Valeur |",
        "|---|---:|",
    ]
    for key, label in METRIC_LABELS.items():
        lines.append(f"| {label} | {_format_metric(metrics.get(key), key)} |")
    lines.extend(
        [
            "",
            "## Rendements annuels",
            "",
            "| Année | Rendement | Période partielle |",
            "|---|---:|:---:|",
        ]
    )
    for item in annual_returns.to_dict(orient="records"):
        lines.append(
            f"| {item['period']} | {item['return']:.2%} | "
            f"{'oui' if item['partial'] else 'non'} |"
        )
    lines.extend(
        [
            "",
            "## Principaux drawdowns",
            "",
            "| Sommet | Creux | Récupération | Drawdown |",
            "|---|---|---|---:|",
        ]
    )
    if drawdowns.empty:
        lines.append("| — | — | — | 0,00 % |")
    for item in drawdowns.to_dict(orient="records"):
        recovered = item["recovery_date"]
        lines.append(
            f"| {item['peak_date'].date().isoformat()} | {item['trough_date'].date().isoformat()} | "
            f"{recovered.date().isoformat() if recovered is not None and not pd.isna(recovered) else 'non récupéré'} | "
            f"{item['drawdown']:.2%} |"
        )
    lines.extend(["", "## Périodes prédéfinies", ""])
    for label, metrics_for_period in analysis["predefined_periods"].items():
        if metrics_for_period is None:
            lines.append(f"- {label} : aucune observation.")
        else:
            lines.append(
                f"- {label} : rendement {_format_metric(metrics_for_period['total_return'], 'total_return')}, "
                f"CAGR {_format_metric(metrics_for_period['cagr'], 'cagr')}, "
                f"drawdown {_format_metric(metrics_for_period['maximum_drawdown'], 'maximum_drawdown')}, "
                f"frais {_format_metric(metrics_for_period['total_fees_eur'], 'total_fees_eur')}."
            )
    lines.extend(
        [
            "",
            "## Frais, rotation et conventions",
            "",
            f"- Frais totaux : {metrics['total_fees_eur']:,.2f} EUR.",
            f"- Notionnel brut échangé : {metrics['total_traded_notional_eur']:,.2f} EUR.",
            f"- Rotation : {metrics['turnover']:.3f}x, définie comme le notionnel brut cumulé divisé par la NAV moyenne.",
            f"- Rendement sans risque fixé à {metrics['risk_free_rate']:.0%}; volatilité/Sharpe annualisés avec {metrics['periods_per_year']} observations par an.",
            "- Le CAGR utilise la fraction d'années calendaires ACT/ACT entre la première et la dernière observation.",
            "- Sharpe utilise l'écart-type échantillonnal des rendements par observation; Sortino utilise le RMS des rendements négatifs avec seuil 0.",
            "- Les ratios non définis (volatilité ou drawdown nul, historique trop court) sont exportés comme null/N/D, jamais comme infini.",
            "",
            "## Graphiques",
            "",
        ]
    )
    for name, filename in payload["charts"].items():
        lines.append(f"- {name}: `./{filename}`")
    lines.extend(
        [
            "",
            "## Conservation du test final et limites",
            "",
            analysis["temporal_split_note"],
            "Les périodes sont descriptives et servent au contrôle du moteur/scénario fixe; elles ne doivent pas servir à sélectionner ou optimiser un futur modèle sur la période de test final.",
            "Le backtest utilise des taux FRED H.10 indicatifs, non certifiés exécutables; disponibilité historique point-in-time inconnue. Les coûts restent paramétriques et ne couvrent pas nécessairement spread, slippage, intérêts ou carry.",
            "",
            "La chronologie complète et la configuration source sont consignées dans `backtest_validation.md`. Résumé de provenance du backtest source :",
            "",
            f"- {metrics['start_date']} – {metrics['end_date']}; NAV initiale {metrics['initial_nav_eur']:,.2f} EUR; NAV finale {metrics['final_nav_eur']:,.2f} EUR.",
            "- Les empreintes SHA-256 des exports, du rapport source et de la configuration sont dans le JSON.",
            "",
        ]
    )
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--nav", type=Path, default=PROJECT_ROOT / "reports" / "backtest_nav.csv")
    parser.add_argument(
        "--transactions",
        type=Path,
        default=PROJECT_ROOT / "reports" / "backtest_transactions.csv",
    )
    parser.add_argument(
        "--allocations",
        type=Path,
        default=PROJECT_ROOT / "reports" / "backtest_allocations.csv",
    )
    parser.add_argument(
        "--backtest-report",
        type=Path,
        default=PROJECT_ROOT / "reports" / "backtest_validation.md",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=PROJECT_ROOT / "configs" / "backtest.yaml",
    )
    parser.add_argument("--output-dir", type=Path, default=PROJECT_ROOT / "reports")
    parser.add_argument("--start-date")
    parser.add_argument("--end-date")
    parser.add_argument("--periods-per-year", type=int, default=252)
    args = parser.parse_args(argv)
    paths = run_performance_analysis(
        nav_path=args.nav,
        transactions_path=args.transactions,
        allocations_path=args.allocations,
        backtest_report_path=args.backtest_report,
        config_path=args.config,
        output_dir=args.output_dir,
        start_date=args.start_date,
        end_date=args.end_date,
        periods_per_year=args.periods_per_year,
    )
    for path in paths.values():
        print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = ["load_backtest_exports", "run_performance_analysis"]
