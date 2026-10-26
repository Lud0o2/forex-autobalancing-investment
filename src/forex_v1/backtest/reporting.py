"""Deterministic CSV, chart, and validation report output for historical runs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import pandas as pd

from forex_v1.backtest.config import BacktestConfig, load_backtest_config
from forex_v1.backtest.data import MarketData, load_market_data
from forex_v1.backtest.engine import BacktestResult, run_backtest
from forex_v1.backtest.strategy import FixedWeightProvider
from forex_v1.data.layout import PROJECT_ROOT


def write_backtest_outputs(
    result: BacktestResult,
    market_data: MarketData,
    output_dir: str | Path,
) -> dict[str, Path]:
    """Write daily NAV, allocations, decisions, transactions, and a capital chart."""

    directory = Path(output_dir)
    directory.mkdir(parents=True, exist_ok=True)
    nav_path = directory / "backtest_nav.csv"
    allocations_path = directory / "backtest_allocations.csv"
    decisions_path = directory / "backtest_decisions.csv"
    transactions_path = directory / "backtest_transactions.csv"
    chart_path = directory / "backtest_capital_curve.png"
    report_path = directory / "backtest_validation.md"

    nav_rows = [
        {
            "market_date": record.market_date.isoformat(),
            "nav_eur": record.nav_eur,
            "cash_eur": record.cash_eur,
        }
        for record in result.daily_records
    ]
    pd.DataFrame(nav_rows, columns=["market_date", "nav_eur", "cash_eur"]).to_csv(
        nav_path,
        index=False,
        float_format="%.12g",
    )

    allocation_rows: list[dict[str, Any]] = []
    for record in result.daily_records:
        prices = market_data.prices_by_date[record.market_date]
        for currency in sorted(record.weights):
            quantity = record.cash_eur if currency == "EUR" else record.units[currency]
            allocation_rows.append(
                {
                    "market_date": record.market_date.isoformat(),
                    "currency": currency,
                    "quantity": quantity,
                    "eur_per_unit": prices[currency],
                    "value_eur": record.values_eur[currency],
                    "weight": record.weights[currency],
                }
            )
    pd.DataFrame(
        allocation_rows,
        columns=["market_date", "currency", "quantity", "eur_per_unit", "value_eur", "weight"],
    ).to_csv(allocations_path, index=False, float_format="%.12g")

    decision_rows = [
        {
            "signal_date": decision.signal_date.isoformat(),
            "information_cutoff_date": decision.information_cutoff_date.isoformat(),
            "information_market_date": (
                decision.information_market_date.isoformat()
                if decision.information_market_date
                else ""
            ),
            "execution_date": decision.execution_date.isoformat() if decision.execution_date else "",
            "status": decision.status,
            "target_weights": (
                json.dumps(decision.target_weights, sort_keys=True, separators=(",", ":"))
                if decision.target_weights is not None
                else ""
            ),
        }
        for decision in result.decisions
    ]
    pd.DataFrame(
        decision_rows,
        columns=[
            "signal_date",
            "information_cutoff_date",
            "information_market_date",
            "execution_date",
            "status",
            "target_weights",
        ],
    ).to_csv(decisions_path, index=False)

    transaction_rows: list[dict[str, Any]] = []
    for execution in result.executions:
        for transaction in execution.rebalance.transactions:
            transaction_rows.append(
                {
                    "signal_date": execution.signal_date.isoformat(),
                    "execution_date": execution.execution_date.isoformat(),
                    "side": transaction.side,
                    "currency": transaction.currency,
                    "quantity": transaction.quantity,
                    "notional_eur": transaction.notional_eur,
                    "fee_eur": transaction.fee_eur,
                }
            )
    pd.DataFrame(
        transaction_rows,
        columns=[
            "signal_date",
            "execution_date",
            "side",
            "currency",
            "quantity",
            "notional_eur",
            "fee_eur",
        ],
    ).to_csv(transactions_path, index=False, float_format="%.12g")

    figure, axis = plt.subplots(figsize=(11, 5))
    axis.plot(
        pd.to_datetime([record.market_date for record in result.daily_records]),
        [record.nav_eur for record in result.daily_records],
        linewidth=1.1,
    )
    axis.set(title="Forex V1 — capital simulé en EUR", xlabel="Date de marché", ylabel="NAV (EUR)")
    axis.grid(True, alpha=0.25)
    figure.tight_layout()
    figure.savefig(chart_path, dpi=140, metadata={"Title": "Forex V1 capital curve"})
    plt.close(figure)

    report_path.write_text(
        _validation_report(result, market_data),
        encoding="utf-8",
    )
    return {
        "nav": nav_path,
        "allocations": allocations_path,
        "decisions": decisions_path,
        "transactions": transactions_path,
        "chart": chart_path,
        "report": report_path,
    }


def _validation_report(result: BacktestResult, market_data: MarketData) -> str:
    first_date = result.daily_records[0].market_date
    last_date = result.daily_records[-1].market_date
    first_executed = result.executions[0] if result.executions else None
    first_actionable = next(
        (decision for decision in result.decisions if decision.target_weights is not None),
        None,
    )
    executed_rebalances = sum(bool(item.rebalance.transactions) for item in result.executions)
    weight_text = ", ".join(
        f"{currency} {weight:.0%}"
        for currency, weight in sorted(result.config.target_weights.items())
        if weight > 0
    )
    return f"""# Validation du backtest historique hebdomadaire

## Scénario et données

- Stratégie fixe illustrative : {weight_text}.
- Capital initial configuré : {result.config.initial_capital_eur:,.2f} EUR.
- Coût de conversion : {result.config.transaction_cost_bps:g} bps par notionnel effectivement converti, soit une fraction de {result.config.transaction_cost_bps / 10_000:g}.
- Aucun intérêt ni carry n'est appliqué aux soldes.
- Source EUR : `{market_data.eur_source_file}` (SHA-256 `{market_data.eur_source_sha256}`).
{f"- Source features : `{market_data.feature_source_file}` (SHA-256 `{market_data.feature_source_sha256}`)." if market_data.feature_source_file else "- Aucune source de features n'a été fournie."}
- Période valorisée : {first_date.isoformat()} à {last_date.isoformat()}, {len(result.daily_records):,} observations communes.

## Chronologie et disponibilité

- Les signaux sont produits chaque vendredi calendaire.
- À un signal T, la dernière observation admissible est la dernière date de marché au plus tard à T−{result.config.availability_lag_days} jours calendaires. Les historiques fournis au provider et son état valorisé utilisent cette date ou une date antérieure.
- La date de marché n'est pas assimilée à la date de publication. La disponibilité réelle des séries FRED reste inconnue ; ce lag est une hypothèse de recherche conservatrice, pas une certification point-in-time.
- L'ordre simulé est exécuté à la première observation de marché strictement postérieure au signal. La fin d'historique sans séance ultérieure est conservée dans le journal des décisions, sans exécution.
- Premier vendredi du calendrier : {result.decisions[0].signal_date.isoformat() if result.decisions else "aucun"}{f" ({result.decisions[0].status})" if result.decisions else ""}.
- Première décision exploitable : {f"{first_actionable.signal_date.isoformat()} (observation d'information {first_actionable.information_market_date.isoformat()})" if first_actionable is not None and first_actionable.information_market_date is not None else "aucune"}.
- Première exécution : {f"{first_executed.execution_date.isoformat()} pour le signal du {first_executed.signal_date.isoformat()}" if first_executed else "aucune"}.

## Résultats

- NAV initiale : {result.initial_nav_eur:,.2f} EUR.
- NAV finale : {result.final_nav_eur:,.2f} EUR.
- Rendement total simple : {result.total_return:.4%}.
- Frais cumulés : {result.total_fees_eur:,.2f} EUR.
- Décisions du vendredi : {len(result.decisions):,}.
- Exécutions : {len(result.executions):,}.
- Rééquilibrages ayant généré au moins une transaction : {executed_rebalances:,}.
- Transactions simulées : {sum(len(item.rebalance.transactions) for item in result.executions):,}.

## Limites

Les prix dérivés des taux FRED H.10 sont des taux de référence indicatifs et ne sont pas des prix exécutables certifiés. Les résultats n'incluent ni spread, slippage, fiscalité, intérêts ou carry. Les conversions sont conceptuellement routées via EUR. Ce résultat est un backtest de recherche, pas une simulation d'ordres réels.

## Fichiers exportés

- `backtest_nav.csv` : valorisation à chaque observation de marché.
- `backtest_allocations.csv` : quantités, valeurs EUR et poids réalisés par date.
- `backtest_decisions.csv` : signal, coupure d'information, observation utilisée et statut d'exécution.
- `backtest_transactions.csv` : détail des montants convertis et frais.
- `backtest_capital_curve.png` : courbe de NAV en EUR.
"""


def run_configured_backtest(
    config_path: str | Path,
    eur_data_path: str | Path,
    output_dir: str | Path,
    feature_data_path: str | Path | None = None,
) -> tuple[BacktestResult, dict[str, Path]]:
    """Load local inputs, execute a run, and write all reproducible report artifacts."""

    config: BacktestConfig = load_backtest_config(config_path)
    market_data = load_market_data(eur_data_path, feature_data_path)
    result = run_backtest(market_data, FixedWeightProvider(config.target_weights), config)
    paths = write_backtest_outputs(result, market_data, output_dir)
    return result, paths


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        default=PROJECT_ROOT / "configs" / "backtest.yaml",
        type=Path,
    )
    parser.add_argument(
        "--eur-data",
        default=PROJECT_ROOT / "data" / "derived" / "eur_rates",
        type=Path,
    )
    parser.add_argument("--feature-data", type=Path)
    parser.add_argument("--reports-dir", default=PROJECT_ROOT / "reports", type=Path)
    arguments = parser.parse_args(argv)
    result, outputs = run_configured_backtest(
        arguments.config,
        arguments.eur_data,
        arguments.reports_dir,
        arguments.feature_data,
    )
    print(
        f"Backtest complete: {result.initial_nav_eur:.2f} EUR -> "
        f"{result.final_nav_eur:.2f} EUR ({result.total_return:.4%}); "
        f"{len(result.executions)} executions, {result.total_fees_eur:.2f} EUR fees."
    )
    for output_path in outputs.values():
        print(output_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
