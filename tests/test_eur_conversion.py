import math
from datetime import UTC, date, datetime

import pytest

from forex_v1.data.eur_conversion import (
    CURRENCY_ORDER,
    build_eur_conversion,
    convert_rates_to_eur,
    read_eur_parquet,
    run_local_conversion,
    select_complete_snapshots,
    write_eur_parquet,
)
from forex_v1.data.providers.fred import FRED_FX_SERIES, FredRawResponse
from forex_v1.data.storage.parquet_store import rebuild_from_raw
from forex_v1.data.storage.raw_store import write_raw_response

SOURCE_RATES = {
    "DEXUSEU": ("1.10", "1.11", "1.20"),
    "DEXUSUK": ("1.25", "1.26", "1.30"),
    "DEXJPUS": ("150", "151", "160"),
    "DEXSZUS": ("0.90", "0.91", "0.95"),
    "DEXCAUS": ("1.30", "1.31", "1.40"),
    "DEXUSAL": ("0.65", "0.66", "0.70"),
    "DEXUSNZ": ("0.60", "0.61", "0.65"),
}
DATES = (date(1999, 1, 4), date(1999, 1, 5), date(2026, 1, 2))
START = date(1999, 1, 1)
FULL_END = date(2026, 12, 31)
PILOT_END = date(2024, 12, 31)


def _write_history(
    root,
    series_id,
    *,
    end_date,
    observation_dates,
    values,
    retrieved_at,
    request_start=START,
):
    payload = {
        "count": str(len(values)),
        "offset": 0,
        "limit": 100000,
        "realtime_start": retrieved_at.date().isoformat(),
        "realtime_end": retrieved_at.date().isoformat(),
        "observations": [
            {"date": market_date.isoformat(), "value": value}
            for market_date, value in zip(observation_dates, values, strict=True)
        ],
    }
    response = FredRawResponse(
        series_id=series_id,
        payload=payload,
        request_params={
            "file_type": "json",
            "limit": 100000,
            "offset": 0,
            "observation_start": request_start.isoformat(),
            "observation_end": end_date.isoformat(),
            "series_id": series_id,
            "sort_order": "asc",
        },
        retrieved_at=retrieved_at,
    )
    raw = write_raw_response(root / "raw", response)
    return rebuild_from_raw(raw.manifest_path, root / "processed")


def _create_source_tree(root):
    selected_window_paths = {}
    for offset, series_id in enumerate(FRED_FX_SERIES):
        rates = SOURCE_RATES[series_id]
        if series_id == "DEXUSUK":
            rates = (rates[0], ".", rates[2])
        selected_window_paths[series_id] = _write_history(
            root,
            series_id,
            end_date=FULL_END,
            observation_dates=DATES,
            values=rates,
            retrieved_at=datetime(2026, 10, 25, 12, 0, offset, tzinfo=UTC),
        )
        _write_history(
            root,
            series_id,
            end_date=date(2025, 12, 31),
            observation_dates=(DATES[0], date(2025, 12, 30)),
            values=(rates[0], rates[1]),
            retrieved_at=datetime(2026, 10, 25, 12, 1, offset, tzinfo=UTC),
        )

    _write_history(
        root,
        "DEXUSEU",
        end_date=PILOT_END,
        observation_dates=(date(2024, 1, 2),),
        values=("1.09",),
        retrieved_at=datetime(2026, 10, 25, 12, 2, tzinfo=UTC),
        request_start=date(2024, 1, 1),
    )
    return selected_window_paths


def test_conversion_formulas_follow_quote_direction():
    rates = {
        "DEXUSEU": 1.10,
        "DEXUSUK": 1.25,
        "DEXJPUS": 150.0,
        "DEXSZUS": 0.90,
        "DEXCAUS": 1.30,
        "DEXUSAL": 0.65,
        "DEXUSNZ": 0.60,
    }

    converted = convert_rates_to_eur(rates)

    assert converted["EUR"] == 1.0
    assert converted["USD"] == pytest.approx(1 / 1.10)
    assert converted["GBP"] == pytest.approx(1.25 / 1.10)
    assert converted["JPY"] == pytest.approx(1 / (150 * 1.10))
    assert converted["CHF"] == pytest.approx(1 / (0.90 * 1.10))
    assert converted["CAD"] == pytest.approx(1 / (1.30 * 1.10))
    assert converted["AUD"] == pytest.approx(0.65 / 1.10)
    assert converted["NZD"] == pytest.approx(0.60 / 1.10)


@pytest.mark.parametrize("invalid", [0.0, -1.0, math.nan, math.inf, -math.inf])
def test_conversion_rejects_non_positive_or_non_finite_inputs(invalid):
    rates = {series_id: 1.0 for series_id in FRED_FX_SERIES}
    rates["DEXUSEU"] = invalid

    with pytest.raises(ValueError, match="finite and strictly positive"):
        convert_rates_to_eur(rates)


def test_conversion_requires_exactly_the_seven_source_series():
    with pytest.raises(ValueError, match="Expected all seven"):
        convert_rates_to_eur({"DEXUSEU": 1.1})


def test_selects_full_history_snapshot_not_2024_pilot(tmp_path):
    expected = _create_source_tree(tmp_path)

    selected = select_complete_snapshots(tmp_path / "processed", tmp_path / "raw")

    assert set(selected) == set(FRED_FX_SERIES)
    for series_id, source in selected.items():
        assert source.parquet.path == expected[series_id].path
        assert source.request_start == START
        assert source.request_end == FULL_END
        assert min(row.market_date for row in source.parquet.observations).year == 1999
        assert max(row.market_date for row in source.parquet.observations).year == 2026


def test_common_calendar_excludes_incomplete_dates_and_marks_long_intervals(tmp_path):
    _create_source_tree(tmp_path)
    sources = select_complete_snapshots(tmp_path / "processed", tmp_path / "raw")

    result = build_eur_conversion(sources)

    assert result.common_dates == (DATES[0], DATES[2])
    assert result.excluded_dates == (DATES[1],)
    assert len(result.rows) == 2 * 8
    assert all(row["availability_known"] is False for row in result.rows)
    assert all(row["available_at"] is None for row in result.rows)
    assert all(
        row["source_market_dates"] == [row["market_date"]] * len(row["source_series_ids"])
        for row in result.rows
    )

    second_day_rows = [row for row in result.rows if row["market_date"] == DATES[2]]
    assert all(
        row["calendar_days_since_previous"] == (DATES[2] - DATES[0]).days for row in second_day_rows
    )
    assert all(row["previous_market_date"] == DATES[0] for row in second_day_rows)


def test_returns_are_adjacent_observation_returns_and_eur_is_stable(tmp_path):
    _create_source_tree(tmp_path)
    result = build_eur_conversion(
        select_complete_snapshots(tmp_path / "processed", tmp_path / "raw")
    )
    first_rows = {row["currency"]: row for row in result.rows[:8]}
    second_rows = {row["currency"]: row for row in result.rows[8:]}

    for currency in CURRENCY_ORDER:
        assert first_rows[currency]["simple_return"] is None
        assert first_rows[currency]["log_return"] is None
    assert second_rows["EUR"]["eur_per_unit"] == 1.0
    assert second_rows["EUR"]["simple_return"] == 0.0
    assert second_rows["EUR"]["log_return"] == 0.0

    for currency in CURRENCY_ORDER[1:]:
        expected_simple = (
            second_rows[currency]["eur_per_unit"] / first_rows[currency]["eur_per_unit"] - 1
        )
        assert second_rows[currency]["simple_return"] == pytest.approx(expected_simple)
        assert second_rows[currency]["log_return"] == pytest.approx(math.log(1 + expected_simple))


def test_derived_parquet_is_idempotent_readable_and_provenance_preserving(tmp_path):
    _create_source_tree(tmp_path)
    result = build_eur_conversion(
        select_complete_snapshots(tmp_path / "processed", tmp_path / "raw")
    )

    first = write_eur_parquet(tmp_path / "derived", result.rows)
    second = write_eur_parquet(tmp_path / "derived", result.rows)
    reloaded = read_eur_parquet(first.path)

    assert first.path == second.path == reloaded.path
    assert first.rows == reloaded.rows == result.rows
    assert reloaded.schema_version == 1
    assert all(row["availability_known"] is False for row in reloaded.rows)
    assert all(
        "DEXUSEU" not in row["source_series_ids"] or row["source_raw_file_ids"]
        for row in reloaded.rows
        if row["currency"] != "EUR"
    )


def test_eur_parquet_date_filter_returns_no_rows_after_cutoff(tmp_path):
    _create_source_tree(tmp_path)
    result = build_eur_conversion(
        select_complete_snapshots(tmp_path / "processed", tmp_path / "raw")
    )
    written = write_eur_parquet(tmp_path / "derived", result.rows)
    cutoff = result.rows[7]["market_date"]

    filtered = read_eur_parquet(written.path, end_date=cutoff)

    assert filtered.rows
    assert max(row["market_date"] for row in filtered.rows) == cutoff
    assert all(row["market_date"] <= cutoff for row in filtered.rows)


def test_local_pipeline_generates_validation_report_and_chart(tmp_path):
    _create_source_tree(tmp_path)

    result = run_local_conversion(
        processed_root=tmp_path / "processed",
        raw_root=tmp_path / "raw",
        derived_root=tmp_path / "derived",
        reports_dir=tmp_path / "reports",
    )

    assert result.derived_path is not None and result.derived_path.is_file()
    report = (tmp_path / "reports" / "eur_conversion_validation.md").read_text(encoding="utf-8")
    chart = tmp_path / "reports" / "eur_conversion_index.png"
    assert "1999-01-04 through 2026-01-02" in report
    assert "Candidate dates excluded" in report
    assert "Unknown" in report or "unknown" in report
    assert chart.is_file() and chart.stat().st_size > 0


def test_invalid_conversion_inputs_do_not_produce_non_finite_outputs():
    rates = {
        "DEXUSEU": 1e-320,
        "DEXUSUK": 1.0,
        "DEXJPUS": 1.0,
        "DEXSZUS": 1.0,
        "DEXCAUS": 1.0,
        "DEXUSAL": 1.0,
        "DEXUSNZ": 1.0,
    }
    with pytest.raises(ValueError, match="Derived EUR rate"):
        convert_rates_to_eur(rates)
