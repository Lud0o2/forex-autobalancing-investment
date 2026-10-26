import json
from datetime import date

import numpy as np
import pandas as pd
import pytest

from forex_v1.evaluation.metrics import analyze_performance, calculate_metrics
from forex_v1.evaluation.reporting import run_performance_analysis
from forex_v1.evaluation.returns import calendar_period_returns
from forex_v1.evaluation.risk import drawdown_episodes


def nav_series(values, dates):
    return pd.Series(values, index=pd.to_datetime(dates), name="nav_eur")


def test_cagr_of_doubling_over_one_actual_calendar_year_is_one_hundred_percent():
    nav = nav_series([100, 200], ["2020-01-01", "2021-01-01"])

    result = calculate_metrics(nav)

    assert result["calendar_years"] == pytest.approx(1.0)
    assert result["cagr"] == pytest.approx(1.0)


def test_drawdown_is_peak_to_trough_not_initial_capital_relative():
    nav = nav_series([100, 120, 90, 110], pd.date_range("2024-01-01", periods=4))

    result = calculate_metrics(nav)

    assert result["maximum_drawdown"] == pytest.approx(-0.25)
    episodes = drawdown_episodes(nav)
    assert episodes.iloc[0]["peak_date"] == pd.Timestamp("2024-01-02")
    assert episodes.iloc[0]["trough_date"] == pd.Timestamp("2024-01-03")
    assert episodes.iloc[0]["recovery_date"] is None


def test_sharpe_and_sortino_use_documented_zero_risk_free_conventions():
    returns = np.array([0.05, 0.02, -0.01])
    levels = [100.0]
    for value in returns:
        levels.append(levels[-1] * (1 + value))
    nav = nav_series(levels, pd.date_range("2024-01-01", periods=4))
    result = calculate_metrics(nav, periods_per_year=252)
    expected_sharpe = returns.mean() / returns.std(ddof=1) * np.sqrt(252)
    downside_rms = np.sqrt((0.0**2 + 0.0**2 + (-0.01) ** 2) / 3)
    expected_sortino = returns.mean() / downside_rms * np.sqrt(252)

    assert result["risk_free_rate"] == 0
    assert result["sharpe_ratio"] == pytest.approx(expected_sharpe)
    assert result["sortino_ratio"] == pytest.approx(expected_sortino)
    assert result["annualized_volatility"] == pytest.approx(returns.std(ddof=1) * np.sqrt(252))


def test_constant_nav_has_zero_returns_volatility_drawdown_and_undefined_ratios():
    nav = nav_series([100, 100, 100], pd.date_range("2024-01-01", periods=3))

    result = calculate_metrics(nav)

    assert result["total_return"] == 0
    assert result["cagr"] == 0
    assert result["maximum_drawdown"] == 0
    assert result["annualized_volatility"] == 0
    assert result["sharpe_ratio"] is None
    assert result["sortino_ratio"] is None
    assert result["calmar_ratio"] is None


def test_single_observation_and_no_transactions_are_handled_without_infinite_ratios():
    nav = nav_series([100], ["2024-01-01"])

    result = calculate_metrics(nav)

    assert result["total_return"] == 0
    assert result["cagr"] is None
    assert result["annualized_volatility"] is None
    assert result["sharpe_ratio"] is None
    assert result["total_fees_eur"] == 0
    assert result["total_traded_notional_eur"] == 0
    assert result["turnover"] == 0


def test_total_fees_and_turnover_are_based_on_gross_executed_notionals():
    nav = nav_series([1000, 1100, 1050], pd.date_range("2024-01-01", periods=3))
    transactions = pd.DataFrame(
        {
            "execution_date": ["2024-01-02", "2024-01-03", "2024-02-01"],
            "notional_eur": [100, 200, 300],
            "fee_eur": [0.1, 0.2, 0.3],
        }
    )

    result = calculate_metrics(nav, transactions=transactions)

    assert result["total_fees_eur"] == pytest.approx(0.3)
    assert result["total_traded_notional_eur"] == pytest.approx(300)
    assert result["turnover"] == pytest.approx(300 / np.mean([1000, 1100, 1050]))


def test_period_returns_flag_truncated_edges_and_use_previous_period_close():
    nav = nav_series(
        [100, 110, 121, 100],
        ["2020-01-15", "2020-12-31", "2021-01-04", "2021-02-15"],
    )

    annual = calendar_period_returns(nav, "year")
    monthly = calendar_period_returns(nav, "month")

    assert annual["return"].tolist() == pytest.approx([0.1, 100 / 110 - 1])
    assert annual["partial"].tolist() == [True, True]
    assert monthly.iloc[0]["partial"]
    assert monthly.iloc[-1]["partial"]


def test_as_of_metric_cutoff_is_invariant_to_future_nav_changes():
    dates = pd.date_range("2024-01-01", periods=6)
    baseline = nav_series([100, 101, 102, 103, 104, 105], dates)
    changed_future = nav_series([100, 101, 102, 103, 1000, 1], dates)

    before = calculate_metrics(baseline, as_of=date(2024, 1, 4))
    after = calculate_metrics(changed_future, as_of=date(2024, 1, 4))

    assert before == after


def test_missing_nav_is_counted_and_dropped_but_infinite_or_nonpositive_is_rejected():
    nav = nav_series([100, np.nan, 110], pd.date_range("2024-01-01", periods=3))

    result = analyze_performance(nav)

    assert result["metrics"]["observation_count"] == 2
    assert result["metrics"]["missing_nav_observations_dropped"] == 1
    with pytest.raises(ValueError, match="finite"):
        calculate_metrics(nav_series([100, np.inf], pd.date_range("2024-01-01", periods=2)))
    with pytest.raises(ValueError, match="strictly positive"):
        calculate_metrics(nav_series([100, 0], pd.date_range("2024-01-01", periods=2)))
    invalid_date_nav = pd.Series([100, 101], index=pd.DatetimeIndex(["2024-01-01", None]))
    with pytest.raises(ValueError, match="dates must be valid"):
        calculate_metrics(invalid_date_nav)


def test_start_and_end_date_selection_excludes_outside_observations():
    nav = nav_series([100, 110, 99, 120], pd.date_range("2024-01-01", periods=4))

    result = calculate_metrics(
        nav,
        start_date="2024-01-02",
        end_date="2024-01-03",
    )

    assert result["start_date"] == "2024-01-02"
    assert result["end_date"] == "2024-01-03"
    assert result["total_return"] == pytest.approx(-0.1)


def test_reporting_reads_exports_and_writes_json_and_six_charts(tmp_path):
    dates = pd.date_range("2024-01-01", periods=3)
    nav_path = tmp_path / "nav.csv"
    transactions_path = tmp_path / "transactions.csv"
    allocations_path = tmp_path / "allocations.csv"
    source_report = tmp_path / "backtest.md"
    config_path = tmp_path / "config.yaml"
    output_path = tmp_path / "out"
    pd.DataFrame(
        {"market_date": dates, "nav_eur": [1000, 1010, 1020], "cash_eur": [1000, 1000, 1000]}
    ).to_csv(nav_path, index=False)
    pd.DataFrame(
        {
            "signal_date": ["2024-01-01"],
            "execution_date": ["2024-01-02"],
            "side": ["buy"],
            "currency": ["USD"],
            "quantity": [12.5],
            "notional_eur": [10],
            "fee_eur": [0.01],
        }
    ).to_csv(transactions_path, index=False)
    allocations = []
    currencies = ["EUR", "USD", "GBP", "JPY", "CHF", "CAD", "AUD", "NZD"]
    for market_date in dates:
        for currency in currencies:
            allocations.append(
                {
                    "market_date": market_date,
                    "currency": currency,
                    "weight": 1.0 if currency == "EUR" else 0.0,
                }
            )
    pd.DataFrame(allocations).to_csv(allocations_path, index=False)
    source_report.write_text("Synthetic source report", encoding="utf-8")
    config_path.write_text(
        "portfolio:\n  initial_capital_eur: 1000\nscenario:\n  transaction_cost_bps: 10\n",
        encoding="utf-8",
    )

    outputs = run_performance_analysis(
        nav_path=nav_path,
        transactions_path=transactions_path,
        allocations_path=allocations_path,
        backtest_report_path=source_report,
        config_path=config_path,
        output_dir=output_path,
    )

    payload = json.loads(outputs["json"].read_text(encoding="utf-8"))
    assert payload["analysis"]["metrics"]["total_fees_eur"] == pytest.approx(0.01)
    assert payload["analysis"]["metrics"]["initial_nav_eur"] == 1000
    assert payload["analysis"]["metrics"]["final_nav_eur"] == 1020
    assert len([key for key in outputs if key.startswith("capital") or key in {"drawdown", "annual", "monthly", "weights", "fees"}]) == 6
    assert all(outputs[key].is_file() for key in ("report", "json", "capital", "drawdown", "annual", "monthly", "weights", "fees"))
