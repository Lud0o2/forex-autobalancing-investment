"""Command-line entry point for explicitly importing FRED Forex history."""

import argparse
import os
import sys
from datetime import UTC, date, datetime, timedelta
from pathlib import Path

from forex_v1.data.coverage import write_coverage_report
from forex_v1.data.importer import (
    ALL_FRED_SERIES,
    SeriesImportResult,
    import_series_batch,
)
from forex_v1.data.providers.fred import (
    FRED_FX_SERIES,
    FredClient,
    FredConfigurationError,
)

PILOT_SERIES = "DEXUSEU"
PILOT_START = date(2024, 1, 1)
PILOT_END = date(2024, 12, 31)
HISTORY_START = date(1999, 1, 1)


def _date_argument(value: str) -> date:
    try:
        return date.fromisoformat(value)
    except ValueError as error:
        raise argparse.ArgumentTypeError("date must use YYYY-MM-DD format") from error


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Import, preserve, and verify FRED H.10 Forex observations."
    )
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument(
        "--dry-run",
        action="store_true",
        help="show the import plan without making network requests",
    )
    mode.add_argument(
        "--execute",
        action="store_true",
        help="authorize real FRED API requests",
    )
    parser.add_argument(
        "--series",
        nargs="+",
        choices=tuple(FRED_FX_SERIES),
        default=None,
        help="FRED series to import (default: DEXUSEU)",
    )
    parser.add_argument("--start-date", type=_date_argument, default=PILOT_START)
    parser.add_argument("--end-date", type=_date_argument, default=PILOT_END)
    parser.add_argument(
        "--full-history",
        action="store_true",
        help="run the required 2024 DEXUSEU pilot, then all seven series from 1999",
    )
    parser.add_argument("--raw-root", type=Path, default=Path("data/raw"))
    parser.add_argument("--processed-root", type=Path, default=Path("data/processed"))
    parser.add_argument("--reports-dir", type=Path, default=Path("reports"))
    return parser


def _report_results(
    results: tuple[SeriesImportResult, ...],
    reports_dir: Path,
) -> None:
    report = write_coverage_report(
        results,
        markdown_path=reports_dir / "data_coverage.md",
        json_path=reports_dir / "data_coverage.json",
    )
    print(
        f"Coverage reports written to {reports_dir / 'data_coverage.md'} "
        f"and {reports_dir / 'data_coverage.json'}"
    )
    common = report["common_history"]
    print(
        "Seven-series common history: "
        f"{common['first_common_date'] or 'N/A'} to {common['last_common_date'] or 'N/A'}; "
        f"{common['common_observation_count']} shared dates"
    )


def _print_results(results: tuple[SeriesImportResult, ...]) -> bool:
    all_succeeded = bool(results)
    for result in results:
        print(f"{result.series_id}: {result.status}")
        if result.raw_snapshot is not None:
            print(
                f"  raw={result.raw_snapshot.raw_file_id} "
                f"sha256={result.raw_snapshot.payload_sha256}"
            )
            metadata = result.raw_snapshot.fred_metadata
            print(
                f"  FRED count={metadata.get('count', 'unknown')} "
                f"offset={metadata.get('offset', 'unknown')} "
                f"limit={metadata.get('limit', 'unknown')}"
            )
        if result.parquet_snapshot is not None:
            print(
                f"  parquet={result.parquet_snapshot.path} "
                f"valid_observations={len(result.parquet_snapshot.observations)}"
            )
            print(
                "  parquet_raw_origin="
                f"{', '.join(result.parquet_snapshot.raw_file_ids)}"
            )
        if result.error:
            print(f"  error={result.error}")
        all_succeeded = all_succeeded and result.status == "success"
    return all_succeeded


def _error_results(
    series_ids: tuple[str, ...],
    start_date: date,
    end_date: date,
    message: str,
) -> tuple[SeriesImportResult, ...]:
    return tuple(
        SeriesImportResult(
            series_id=series_id,
            status="error",
            start_date=start_date,
            end_date=end_date,
            error=message,
        )
        for series_id in series_ids
    )


def _run_full_history(
    client: FredClient,
    *,
    raw_root: Path,
    processed_root: Path,
) -> tuple[tuple[SeriesImportResult, ...], tuple[SeriesImportResult, ...]]:
    """Run the specified small pilot first and proceed only after it succeeds."""
    pilot = import_series_batch(
        client,
        (PILOT_SERIES,),
        start_date=PILOT_START,
        end_date=PILOT_END,
        raw_root=raw_root,
        processed_root=processed_root,
    )
    if not pilot or pilot[0].status != "success":
        return pilot, pilot

    last_complete_day = datetime.now(UTC).date() - timedelta(days=1)
    full_results = import_series_batch(
        client,
        ALL_FRED_SERIES,
        start_date=HISTORY_START,
        end_date=last_complete_day,
        raw_root=raw_root,
        processed_root=processed_root,
    )
    return pilot, full_results


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.start_date > args.end_date:
        _parser().error("--start-date must not be after --end-date")
    if args.full_history and args.series is not None:
        _parser().error("--series cannot be combined with --full-history")

    if args.dry_run:
        if args.full_history:
            print(
                "DRY RUN: no network requests will be made. Planned sequence: "
                "DEXUSEU 2024-01-01..2024-12-31 pilot, then all seven series "
                f"{HISTORY_START.isoformat()}..{datetime.now(UTC).date() - timedelta(days=1)}."
            )
            planned_series = (PILOT_SERIES,)
            start_date, end_date = PILOT_START, PILOT_END
        else:
            planned_series = tuple(args.series or (PILOT_SERIES,))
            start_date, end_date = args.start_date, args.end_date
            print(
                "DRY RUN: no network requests will be made. Planned import: "
                f"{', '.join(planned_series)} from {start_date} through {end_date}."
            )
        results = tuple(
            SeriesImportResult(
                series_id=series_id,
                status="dry_run",
                start_date=start_date,
                end_date=end_date,
                error="No network request was made.",
            )
            for series_id in planned_series
        )
        _report_results(results, args.reports_dir)
        return 0

    api_key = os.environ.get("FRED_API_KEY")
    if not api_key or not api_key.strip():
        message = (
            "FRED_API_KEY is not configured. Set it in the process environment, "
            "then rerun --execute. The key value is never displayed or stored."
        )
        print(message, file=sys.stderr)
        if args.full_history:
            results = _error_results(
                (PILOT_SERIES,), PILOT_START, PILOT_END, "FRED_API_KEY is missing."
            )
        else:
            series_ids = tuple(args.series or (PILOT_SERIES,))
            results = _error_results(series_ids, args.start_date, args.end_date, message)
        _report_results(results, args.reports_dir)
        _print_results(results)
        return 2

    client = FredClient()
    if args.full_history:
        pilot_results, results = _run_full_history(
            client,
            raw_root=args.raw_root,
            processed_root=args.processed_root,
        )
        print("2024 DEXUSEU pilot:")
        pilot_succeeded = _print_results(pilot_results)
        if pilot_succeeded:
            print("Full-history import:")
        else:
            print("Full-history import skipped because the pilot did not succeed.")
    else:
        series_ids = tuple(args.series or (PILOT_SERIES,))
        results = import_series_batch(
            client,
            series_ids,
            start_date=args.start_date,
            end_date=args.end_date,
            raw_root=args.raw_root,
            processed_root=args.processed_root,
        )

    _report_results(results, args.reports_dir)
    return 0 if _print_results(results) else 1


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except FredConfigurationError as error:
        print(str(error), file=sys.stderr)
        raise SystemExit(2) from error
