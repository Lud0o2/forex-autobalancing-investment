"""Quality summary and common-history reporting for imported FRED observations."""

import json
import math
from collections import Counter
from datetime import date, timedelta
from pathlib import Path
from typing import Any

from forex_v1.data.importer import SeriesImportResult
from forex_v1.data.providers.fred import FRED_FX_SERIES
from forex_v1.data.storage.parquet_store import read_parquet
from forex_v1.data.storage.raw_store import read_raw_response


def _parse_valid_value(value: Any) -> tuple[bool, bool]:
    if value in (".", ""):
        return False, True
    try:
        parsed = float(value)
    except (TypeError, ValueError, OverflowError):
        return False, False
    return math.isfinite(parsed) and parsed > 0, False


def _longest_weekday_gap(valid_dates: set[date]) -> int:
    if len(valid_dates) < 2:
        return 0
    current = min(valid_dates)
    final = max(valid_dates)
    longest = run = 0
    while current <= final:
        if current.weekday() < 5:
            if current not in valid_dates:
                run += 1
                longest = max(longest, run)
            else:
                run = 0
        current += timedelta(days=1)
    return longest


def _series_report(result: SeriesImportResult) -> dict[str, Any]:
    base_currency, quote_currency = FRED_FX_SERIES[result.series_id]
    report: dict[str, Any] = {
        "series_id": result.series_id,
        "base_currency": base_currency,
        "quote_currency": quote_currency,
        "status": result.status,
        "requested_start_date": result.start_date.isoformat(),
        "requested_end_date": result.end_date.isoformat(),
        "first_valid_date": None,
        "last_valid_date": None,
        "valid_observation_count": None,
        "explicit_missing_value_count": None,
        "total_dates_returned": None,
        "coverage_percent": None,
        "weekdays_without_observation": None,
        "longest_weekday_interruption": None,
        "duplicate_dates": None,
        "anomalous_prices": None,
        "invalid_dates": None,
        "raw_file_id": None,
        "raw_sha256": None,
        "parquet_file": None,
        "parquet_raw_file_ids": None,
        "parquet_schema_version": None,
        "availability_known": False,
        "error": result.error,
    }
    if result.raw_snapshot is None:
        return report

    raw = read_raw_response(result.raw_snapshot.manifest_path)
    payload_observations = raw.payload["observations"]
    valid_dates: list[date] = []
    missing_values = anomalies = invalid_dates = 0
    dates: list[date] = []
    for item in payload_observations:
        if not isinstance(item, dict) or not isinstance(item.get("date"), str):
            invalid_dates += 1
            continue
        try:
            item_date = date.fromisoformat(item["date"])
        except ValueError:
            invalid_dates += 1
            continue
        dates.append(item_date)
        is_valid, is_missing = _parse_valid_value(item.get("value"))
        if is_valid:
            valid_dates.append(item_date)
        elif is_missing:
            missing_values += 1
        else:
            anomalies += 1

    counts = Counter(dates)
    valid_date_set = set(valid_dates)
    valid_sorted = sorted(valid_dates)
    report.update(
        {
            "first_valid_date": valid_sorted[0].isoformat() if valid_sorted else None,
            "last_valid_date": valid_sorted[-1].isoformat() if valid_sorted else None,
            "valid_observation_count": len(valid_dates),
            "explicit_missing_value_count": missing_values,
            "total_dates_returned": len(payload_observations),
            "coverage_percent": (
                round(len(valid_dates) / len(payload_observations) * 100, 2)
                if payload_observations
                else 0.0
            ),
            "weekdays_without_observation": sum(
                1
                for day in _date_range(valid_sorted[0], valid_sorted[-1])
                if day.weekday() < 5 and day not in valid_date_set
            )
            if valid_sorted
            else 0,
            "longest_weekday_interruption": _longest_weekday_gap(valid_date_set),
            "duplicate_dates": sum(count - 1 for count in counts.values() if count > 1),
            "anomalous_prices": anomalies,
            "invalid_dates": invalid_dates,
            "raw_file_id": raw.raw_file_id,
            "raw_sha256": raw.payload_sha256,
            "parquet_file": (
                str(result.parquet_snapshot.path) if result.parquet_snapshot is not None else None
            ),
            "parquet_raw_file_ids": (
                list(result.parquet_snapshot.raw_file_ids)
                if result.parquet_snapshot is not None
                else None
            ),
            "parquet_schema_version": (
                result.parquet_snapshot.schema_version
                if result.parquet_snapshot is not None
                else None
            ),
        }
    )
    return report


def _date_range(start: date, end: date):
    current = start
    while current <= end:
        yield current
        current += timedelta(days=1)


def build_coverage_report(results: tuple[SeriesImportResult, ...]) -> dict[str, Any]:
    """Calculate per-series coverage and date intersection without filling gaps."""
    by_series = {result.series_id: result for result in results}
    series_reports = [
        _series_report(by_series[series_id])
        if series_id in by_series
        else {
            "series_id": series_id,
            "base_currency": FRED_FX_SERIES[series_id][0],
            "quote_currency": FRED_FX_SERIES[series_id][1],
            "status": "not_attempted",
            "error": "Series was not part of this import run.",
        }
        for series_id in FRED_FX_SERIES
    ]
    successful = [row for row in series_reports if row["status"] == "success"]
    common_history: dict[str, Any] = {
        "first_common_date": None,
        "last_common_date": None,
        "common_observation_count": 0,
        "coverage_denominator": None,
        "coverage_denominator_definition": (
            "Distinct valid market dates in the union of all series, within the common-date span."
        ),
        "coverage_by_series": {},
    }
    if len(successful) == len(FRED_FX_SERIES):
        observations_by_series = {}
        for result in results:
            if result.status == "success" and result.parquet_snapshot is not None:
                parquet = read_parquet(result.parquet_snapshot.path)
                observations_by_series[result.series_id] = {
                    observation.market_date for observation in parquet.observations
                }
        shared_dates = set.intersection(*observations_by_series.values())
        if shared_dates:
            first_common = min(shared_dates)
            last_common = max(shared_dates)
            range_union = set.union(*(dates for dates in observations_by_series.values() if dates))
            common_period_union = {
                market_date
                for market_date in range_union
                if first_common <= market_date <= last_common
            }
            common_history.update(
                {
                    "first_common_date": first_common.isoformat(),
                    "last_common_date": last_common.isoformat(),
                    "common_observation_count": len(shared_dates),
                    "coverage_denominator": len(common_period_union),
                    "coverage_by_series": {
                        series_id: len(dates & common_period_union) / len(common_period_union)
                        if common_period_union
                        else None
                        for series_id, dates in observations_by_series.items()
                    },
                }
            )

    return {
        "report_version": 1,
        "point_in_time_certified": False,
        "availability_note": (
            "FRED H.10 publication availability is not historically certified. "
            "available_at remains unknown; these observations are not a point-in-time "
            "certified basis for executable decisions."
        ),
        "absence_note": (
            "Weekday gaps are descriptive only. Weekends are excluded from weekday-gap "
            "counts; public holidays and true coverage anomalies cannot be reliably "
            "distinguished from this response alone."
        ),
        "series": series_reports,
        "common_history": common_history,
    }


def write_coverage_report(
    results: tuple[SeriesImportResult, ...],
    *,
    markdown_path: str | Path,
    json_path: str | Path,
) -> dict[str, Any]:
    """Write a machine-readable and human-readable coverage report."""
    report = build_coverage_report(results)
    json_output = Path(json_path)
    markdown_output = Path(markdown_path)
    json_output.parent.mkdir(parents=True, exist_ok=True)
    markdown_output.parent.mkdir(parents=True, exist_ok=True)
    json_output.write_text(
        json.dumps(report, indent=2, sort_keys=True, ensure_ascii=False),
        encoding="utf-8",
    )

    lines = [
        "# FRED data coverage report",
        "",
        report["availability_note"],
        "",
        report["absence_note"],
        "",
        "| Series | Pair | Status | First valid | Last valid | Valid | Missing | Dates returned | Coverage % | Weekday gaps | Duplicate dates | Price anomalies | Invalid dates |",
        "| --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in report["series"]:
        lines.append(
            "| {series_id} | {base_currency}/{quote_currency} | {status} | {first_valid_date} "
            "| {last_valid_date} | {valid_observation_count} | {explicit_missing_value_count} "
            "| {total_dates_returned} | {coverage_percent} | {weekdays_without_observation} | {duplicate_dates} "
            "| {anomalous_prices} | {invalid_dates} |".format(
                **{
                    key: row.get(key) if row.get(key) is not None else "N/A"
                    for key in (
                        "series_id",
                        "base_currency",
                        "quote_currency",
                        "status",
                        "first_valid_date",
                        "last_valid_date",
                        "valid_observation_count",
                        "explicit_missing_value_count",
                        "total_dates_returned",
                        "coverage_percent",
                        "weekdays_without_observation",
                        "duplicate_dates",
                        "anomalous_prices",
                        "invalid_dates",
                    )
                }
            )
        )
    common = report["common_history"]
    lines.extend(
        [
            "",
            "## Common history",
            "",
            f"- First common date: {common['first_common_date'] or 'N/A'}",
            f"- Last common date: {common['last_common_date'] or 'N/A'}",
            f"- Dates with all seven series: {common['common_observation_count']}",
            f"- Coverage denominator: {common['coverage_denominator']}",
            "",
            "No forward-fill or interpolation is applied.",
            "",
        ]
    )
    markdown_output.write_text("\n".join(lines), encoding="utf-8")
    return report
