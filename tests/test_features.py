import math
from datetime import date, timedelta

import numpy as np
import pandas as pd
import pytest

from forex_v1.data.eur_conversion import (
    CURRENCY_ORDER,
    SERIES_BY_CURRENCY,
    EurParquetSnapshot,
    write_eur_parquet,
)
from forex_v1.features.builder import (
    FeatureStoreError,
    build_feature_records,
    load_feature_configuration,
    read_feature_parquet,
    run_feature_pipeline,
    summarize_features,
    write_feature_parquet,
)
from forex_v1.features.technical import (
    calculate_bollinger_zscore,
    calculate_channel_position,
    calculate_drawdown,
    calculate_log_return,
    calculate_momentum,
    calculate_rsi,
    calculate_sma_relative,
    calculate_technical_features,
    calculate_volatility,
)
from forex_v1.features.validation import validate_feature_configuration


def make_prices(values):
    return pd.Series(
        values,
        index=pd.date_range("2020-01-01", periods=len(values), freq="B"),
        dtype=float,
    )


def minimal_configuration():
    return {
        "version": 1,
        "annualization_observations": 252,
        "indicators": {
            "log_return": {"enabled": True, "windows": [1, 5]},
            "rsi": {"enabled": True, "windows": [3]},
            "momentum": {"enabled": True, "windows": [5]},
            "sma_relative": {"enabled": True, "windows": [3]},
            "volatility": {"enabled": True, "windows": [3]},
            "bollinger_zscore": {"enabled": True, "windows": [3]},
            "channel_position": {"enabled": True, "windows": [3]},
            "drawdown": {"enabled": True, "windows": [3]},
        },
    }


def make_eur_snapshot(tmp_path, *, rows=250, omit_currency=None):
    dates = [date(2020, 1, 1) + timedelta(days=index) for index in range(rows)]
    currencies = CURRENCY_ORDER
    data = []
    values_by_currency = {}
    for currency_index, currency in enumerate(currencies):
        values_by_currency[currency] = []
        for index, market_date in enumerate(dates):
            if currency == "EUR":
                value = 1.0
            else:
                foreign_index = currency_index - 1
                value = (0.5 + foreign_index * 0.1) * math.exp(
                    0.0002 * index + 0.003 * math.sin(index / 7 + foreign_index)
                )
            values_by_currency[currency].append(value)
    for market_date_index, market_date in enumerate(dates):
        for currency in currencies:
            if omit_currency == currency and market_date_index == rows // 2:
                continue
            value = values_by_currency[currency][market_date_index]
            series_ids = list(SERIES_BY_CURRENCY[currency])
            previous_date = dates[market_date_index - 1] if market_date_index else None
            previous_value = (
                values_by_currency[currency][market_date_index - 1]
                if market_date_index
                else None
            )
            simple_return = (
                value / previous_value - 1 if previous_value is not None else None
            )
            log_return = (
                math.log(value / previous_value) if previous_value is not None else None
            )
            data.append(
                {
                    "market_date": market_date,
                    "currency": currency,
                    "eur_per_unit": value,
                    "simple_return": simple_return,
                    "log_return": log_return,
                    "previous_market_date": previous_date,
                    "calendar_days_since_previous": (
                        (market_date - previous_date).days if previous_date else None
                    ),
                    "source_series_ids": series_ids,
                    "source_raw_file_ids": [f"fred/{series_id}/source.json" for series_id in series_ids],
                    "source_parquet_files": [f"{series_id}.parquet" for series_id in series_ids],
                    "source_market_dates": [market_date for _ in series_ids],
                    "available_at": None,
                    "availability_known": False,
                }
            )
    source_path = tmp_path / "eur_source.parquet"
    source_path.write_bytes(b"fixture EUR source parquet identity")
    return EurParquetSnapshot(
        rows=tuple(data),
        transform_version="1",
        schema_version=1,
        path=source_path,
    )


def test_rsi_up_down_flat_and_zero_range_conventions():
    up = calculate_rsi(make_prices(range(1, 16)), window=14)
    down = calculate_rsi(make_prices(range(15, 0, -1)), window=14)
    flat = calculate_rsi(make_prices([2.0] * 16), window=14)

    assert up.iloc[:14].isna().all()
    assert up.iloc[14] == pytest.approx(1.0)
    assert down.iloc[14] == pytest.approx(0.0)
    assert flat.iloc[14] == pytest.approx(0.5)
    assert flat.dropna().between(0, 1).all()


@pytest.mark.parametrize(
    ("values", "expected_sign"),
    [([1, 2, 3, 4, 5, 6], 1), ([6, 5, 4, 3, 2, 1], -1), ([2] * 6, 0)],
)
def test_momentum_has_expected_sign(values, expected_sign):
    momentum = calculate_momentum(make_prices(values), window=5).iloc[-1]
    assert np.sign(momentum) == expected_sign
    assert momentum == pytest.approx(math.log(values[-1] / values[0]))


def test_sma_relative_and_lookback_are_observation_based():
    prices = make_prices([1, 2, 3, 4])
    values = calculate_sma_relative(prices, window=3)
    assert values.iloc[:2].isna().all()
    assert values.iloc[2] == pytest.approx(3 / 2 - 1)
    assert values.iloc[3] == pytest.approx(4 / 3 - 1)
    assert calculate_log_return(prices, window=2).iloc[2] == pytest.approx(math.log(3 / 1))


def test_volatility_uses_sample_std_of_log_returns_and_annualization():
    prices = make_prices([1, 2, 4, 8])
    result = calculate_volatility(prices, window=3, annualization_observations=252)
    expected_returns = np.log(np.array([2.0, 2.0, 2.0]))
    assert result.iloc[:3].isna().all()
    assert result.iloc[3] == pytest.approx(np.std(expected_returns, ddof=1) * math.sqrt(252))


def test_bollinger_zscore_channel_and_drawdown_manual_values():
    prices = make_prices([1, 2, 3, 2])
    zscore = calculate_bollinger_zscore(prices, window=3)
    channel = calculate_channel_position(prices, window=3)
    drawdown = calculate_drawdown(prices, window=3)

    assert zscore.iloc[2] == pytest.approx((3 - 2) / np.std([1, 2, 3], ddof=0))
    assert channel.iloc[2] == pytest.approx(1.0)
    assert drawdown.iloc[3] == pytest.approx(2 / 3 - 1)
    assert channel.dropna().between(0, 1).all()


def test_flat_series_and_short_warmup_windows_are_explicit():
    prices = make_prices([5.0] * 8)
    zscore = calculate_bollinger_zscore(prices, window=3)
    channel = calculate_channel_position(prices, window=3)
    volatility = calculate_volatility(prices, window=3)
    drawdown = calculate_drawdown(prices, window=3)

    assert zscore.iloc[:2].isna().all()
    assert channel.iloc[:2].isna().all()
    assert zscore.iloc[2:].eq(0).all()
    assert channel.iloc[2:].eq(0.5).all()
    assert volatility.iloc[3:].eq(0).all()
    assert drawdown.iloc[2:].eq(0).all()


def test_config_loader_and_indicator_selection(tmp_path):
    configuration_path = tmp_path / "features.yaml"
    configuration_path.write_text(
        """
version: 7
annualization_observations: 252
indicators:
  momentum:
    enabled: true
    windows: [5]
  rsi:
    enabled: false
    windows: [14]
""",
        encoding="utf-8",
    )
    configuration, digest = load_feature_configuration(configuration_path)
    output = calculate_technical_features(make_prices(range(1, 10)), configuration)

    assert digest
    assert list(output.columns) == ["momentum_5"]
    assert output["momentum_5"].iloc[:5].isna().all()
    assert output["momentum_5"].iloc[-1] == pytest.approx(math.log(9 / 4))


@pytest.mark.parametrize(
    "configuration",
    [
        {},
        {"version": 1, "annualization_observations": 0, "indicators": {}},
        {
            "version": 1,
            "annualization_observations": 252,
            "indicators": {"atr": {"enabled": True, "windows": [14]}},
        },
        {
            "version": 1,
            "annualization_observations": 252,
            "indicators": {"rsi": {"enabled": True, "windows": [0]}},
        },
    ],
)
def test_invalid_indicator_configuration_is_rejected(configuration):
    with pytest.raises(ValueError):
        validate_feature_configuration(configuration)


def test_indicators_are_invariant_to_positive_price_scale():
    prices = make_prices([100 + math.sin(index / 4) * 7 + index / 10 for index in range(250)])
    configuration = minimal_configuration()
    base = calculate_technical_features(prices, configuration)
    scaled = calculate_technical_features(prices * 17.5, configuration)

    pd.testing.assert_frame_equal(base, scaled, check_exact=False, rtol=1e-9, atol=1e-9)


def test_no_feature_before_modified_prices_changes():
    prices = make_prices([100 + index * 0.1 + math.sin(index / 5) for index in range(400)])
    configuration = minimal_configuration()
    original = calculate_technical_features(prices, configuration)
    changed = prices.copy()
    changed.iloc[-30:] *= 1000
    recalculated = calculate_technical_features(changed, configuration)

    pd.testing.assert_frame_equal(
        original.iloc[:-30],
        recalculated.iloc[:-30],
        check_exact=True,
    )


def test_features_are_comparable_across_currency_price_scales():
    prices = make_prices([1 + index * 0.02 + math.sin(index / 3) for index in range(50)])
    configuration = minimal_configuration()
    base = calculate_technical_features(prices, configuration)
    scaled = calculate_technical_features(prices * 1000, configuration)

    pd.testing.assert_frame_equal(base, scaled, check_exact=False, rtol=1e-9, atol=1e-9)


def test_build_preserves_dates_provenance_and_missing_values(tmp_path):
    configuration = minimal_configuration()
    snapshot = make_eur_snapshot(tmp_path, rows=250)
    records = build_feature_records(snapshot, configuration, configuration_sha256="abc123")

    assert len(records) == 250 * 7
    row = next(
        row
        for row in records
        if row["market_date"] == date(2020, 1, 1) and row["currency"] == "USD"
    )
    assert row["currency"] == "USD"
    assert row["source_eur_parquet"] == snapshot.path.name
    assert row["source_eur_sha256"]
    assert row["source_raw_file_ids"]
    assert row["feature_config_version"] == "1"
    assert row["feature_config_sha256"] == "abc123"
    assert row["availability_known"] is False
    assert row["available_at"] is None
    assert row["missing_feature_count"] > 0
    assert row["all_features_available"] is False


def test_misaligned_currency_calendar_is_rejected(tmp_path):
    snapshot = make_eur_snapshot(tmp_path, rows=30, omit_currency="CAD")

    with pytest.raises(FeatureStoreError, match="common calendar"):
        build_feature_records(
            snapshot,
            minimal_configuration(),
            configuration_sha256="hash",
        )


def test_feature_parquet_round_trip_is_idempotent_and_validated(tmp_path):
    snapshot = make_eur_snapshot(tmp_path, rows=250)
    records = build_feature_records(
        snapshot,
        minimal_configuration(),
        configuration_sha256="configuration-hash",
    )

    first = write_feature_parquet(tmp_path / "derived", records)
    second = write_feature_parquet(tmp_path / "derived", records)
    loaded = read_feature_parquet(first)

    assert first == second == loaded.path
    assert loaded.records == records
    assert loaded.configuration_sha256 == "configuration-hash"
    assert loaded.feature_names == tuple(sorted(name for name in records[0] if name.endswith(("_1", "_3", "_5"))))
    assert loaded.records[0]["market_date"] == records[0]["market_date"]


def test_feature_pipeline_reads_source_and_reports_warmup_summary(tmp_path):
    snapshot = make_eur_snapshot(tmp_path, rows=250)
    source_file = write_eur_parquet(tmp_path / "input", snapshot.rows)
    config_path = tmp_path / "features.yaml"
    config_path.write_text(
        """
version: 1
annualization_observations: 252
indicators:
  sma_relative:
    enabled: true
    windows: [20]
""",
        encoding="utf-8",
    )

    output = run_feature_pipeline(
        eur_parquet_path=source_file.path,
        configuration_path=config_path,
        output_root=tmp_path / "data" / "derived",
        report_path=tmp_path / "reports" / "features_validation.md",
    )
    summary = summarize_features(output)

    assert summary["row_count"] == 250 * 7
    assert summary["first_complete_date_all_currencies"] == date(2020, 1, 20)
    assert summary["feature_columns"]["sma_relative_20"]["missing_count"] == 19 * 7
    assert (tmp_path / "reports" / "features_validation.md").is_file()
