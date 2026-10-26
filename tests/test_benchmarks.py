import json
from datetime import date, timedelta

import pandas as pd

from forex_v1.backtest.data import MarketData
from forex_v1.benchmarks import run_benchmark_comparison
from forex_v1.data.schemas import SUPPORTED_CURRENCIES
from forex_v1.strategies import BENCHMARK_NAMES


def synthetic_market_data(*, change_after: date | None = None):
    dates = []
    current = date(2020, 1, 1)
    while len(dates) < 65:
        if current.weekday() < 5:
            dates.append(current)
        current += timedelta(days=1)
    prices = {}
    features = {}
    currencies = sorted(SUPPORTED_CURRENCIES - {"EUR"})
    for index, market_date in enumerate(dates):
        changed = change_after is not None and market_date > change_after
        prices[market_date] = {
            "EUR": 1.0,
            "USD": 0.8 + (0.02 * index if changed else 0.001 * index),
            "GBP": 1.1,
            "JPY": 0.006,
            "CHF": 1.0,
            "CAD": 0.7,
            "AUD": 0.65,
            "NZD": 0.6,
        }
        features[market_date] = {
            **{
                f"{currency}__log_return_60": (
                    (-1) ** (index + offset) * 0.02 + (0.01 if changed else 0.0)
                )
                for offset, currency in enumerate(currencies)
            },
            **{
                f"{currency}__volatility_60": 0.05 + offset * 0.01
                for offset, currency in enumerate(currencies)
            },
        }
    return MarketData(
        prices_by_date=prices,
        eur_source_file="synthetic-eur.parquet",
        eur_source_sha256="synthetic-eur-sha256",
        features_by_date=features,
        feature_names=("log_return_60", "volatility_60"),
        complete_feature_dates=frozenset(dates),
    )


def configuration():
    return {
        "initial_capital_eur": 10_000.0,
        "transaction_cost_bps": 10.0,
        "availability_lag_days": 7,
        "development_fraction": 0.75,
        "max_foreign_weight": 0.35,
        "strategies": BENCHMARK_NAMES,
    }


def test_comparison_uses_one_development_nav_signal_and_execution_calendar(tmp_path):
    data = synthetic_market_data()

    result = run_benchmark_comparison(
        data,
        configuration=configuration(),
        output_dir=tmp_path,
    )

    assert set(result["strategies"]) == set(BENCHMARK_NAMES)
    assert all(item["status"] == "completed" for item in result["strategies"].values())
    assert result["period"]["first_complete_feature_date"] == "2020-01-01"
    assert result["period"]["final_test_period_evaluated"] is False
    assert result["comparison_validation"]["common_nav_dates"] is True
    assert result["comparison_validation"]["common_signal_dates"] is True
    assert result["comparison_validation"]["common_weekly_execution_dates"] is True
    assert result["strategies"]["buy_hold_equal_weight"]["execution_count"] == 1
    assert result["strategies"]["eur_cash"]["transaction_count"] == 0
    assert result["strategies"]["inverse_volatility_60"]["metrics"]["maximum_drawdown"] <= 0
    assert (tmp_path / "benchmark_comparison.md").is_file()
    assert (tmp_path / "benchmark_comparison.json").is_file()
    for name in BENCHMARK_NAMES:
        folder = tmp_path / "benchmarks" / name
        assert {path.name for path in folder.iterdir()} == {
            "nav.csv",
            "allocations.csv",
            "decisions.csv",
            "transactions.csv",
        }


def test_future_market_and_features_do_not_change_development_decisions_or_metrics(tmp_path):
    baseline = synthetic_market_data()
    boundary_index = int(len(baseline.dates) * 0.75) - 1
    development_end = baseline.dates[boundary_index]
    changed = synthetic_market_data(change_after=development_end)

    first = run_benchmark_comparison(
        baseline,
        configuration=configuration(),
        output_dir=tmp_path / "baseline",
    )
    second = run_benchmark_comparison(
        changed,
        configuration=configuration(),
        output_dir=tmp_path / "changed",
    )

    assert first["period"] == second["period"]
    for name in BENCHMARK_NAMES:
        assert first["strategies"][name]["metrics"] == second["strategies"][name]["metrics"]
        pd.testing.assert_frame_equal(
            pd.read_csv(tmp_path / "baseline" / "benchmarks" / name / "decisions.csv"),
            pd.read_csv(tmp_path / "changed" / "benchmarks" / name / "decisions.csv"),
        )


def test_weekly_and_buy_hold_exports_are_reproducible(tmp_path):
    data = synthetic_market_data()

    first = run_benchmark_comparison(
        data,
        configuration=configuration(),
        output_dir=tmp_path / "first",
    )
    second = run_benchmark_comparison(
        data,
        configuration=configuration(),
        output_dir=tmp_path / "second",
    )

    assert first["strategies"] == second["strategies"]
    for name in BENCHMARK_NAMES:
        for filename in ("nav.csv", "allocations.csv", "decisions.csv", "transactions.csv"):
            assert (
                tmp_path / "first" / "benchmarks" / name / filename
            ).read_bytes() == (
                tmp_path / "second" / "benchmarks" / name / filename
            ).read_bytes()
    payload = json.loads((tmp_path / "first" / "benchmark_comparison.json").read_text())
    assert payload["period"]["final_test_period_evaluated"] is False
