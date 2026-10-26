"""Financial performance metrics with explicit research conventions."""

from __future__ import annotations

import math
from datetime import date, datetime
from typing import Any

import numpy as np
import pandas as pd

from forex_v1.evaluation.returns import calendar_period_returns, observation_returns
from forex_v1.evaluation.risk import drawdown_series, maximum_drawdown

SECONDS_PER_DAY = 86_400


def _calendar_year_fraction(start: pd.Timestamp, end: pd.Timestamp) -> float:
    """Calculate ACT/ACT elapsed calendar years, respecting leap years."""

    if end <= start:
        return 0.0
    cursor = start
    fraction = 0.0
    while cursor.year < end.year:
        next_year = pd.Timestamp(year=cursor.year + 1, month=1, day=1, tz=cursor.tz)
        fraction += (next_year - cursor).total_seconds() / (
            SECONDS_PER_DAY * (366 if _is_leap_year(cursor.year) else 365)
        )
        cursor = next_year
    fraction += (end - cursor).total_seconds() / (
        SECONDS_PER_DAY * (366 if _is_leap_year(cursor.year) else 365)
    )
    return fraction


def _is_leap_year(year: int) -> bool:
    return year % 4 == 0 and (year % 100 != 0 or year % 400 == 0)


def validate_nav(
    nav: pd.Series,
    *,
    start_date: date | datetime | str | None = None,
    end_date: date | datetime | str | None = None,
    as_of: date | datetime | str | None = None,
) -> tuple[pd.Series, int]:
    """Validate, sort, optionally truncate, and explicitly count missing NAV values."""

    if not isinstance(nav, pd.Series):
        raise TypeError("nav must be a pandas Series")
    if not isinstance(nav.index, pd.DatetimeIndex):
        try:
            index = pd.to_datetime(nav.index, errors="raise")
        except (TypeError, ValueError) as error:
            raise ValueError("NAV index must contain valid dates") from error
        nav = nav.copy()
        nav.index = pd.DatetimeIndex(index)
    if bool(pd.isna(nav.index).any()):
        raise ValueError("NAV dates must be valid")
    if nav.index.has_duplicates:
        raise ValueError("NAV dates must be unique")
    nav = nav.sort_index().astype(float)
    nav_dates = pd.DatetimeIndex(nav.index)
    if start_date is not None or end_date is not None or as_of is not None:
        lower = pd.Timestamp(start_date) if start_date is not None else None
        upper_value = end_date if end_date is not None else as_of
        cutoff = pd.Timestamp(upper_value) if upper_value is not None else None
        if nav_dates.tz is not None:
            if cutoff is not None and cutoff.tzinfo is None:
                cutoff = cutoff.tz_localize(nav_dates.tz)
            if lower is not None and lower.tzinfo is None:
                lower = lower.tz_localize(nav_dates.tz)
        elif nav_dates.tz is None:
            if cutoff is not None and cutoff.tzinfo is not None:
                cutoff = cutoff.tz_localize(None)
            if lower is not None and lower.tzinfo is not None:
                lower = lower.tz_localize(None)
        if lower is not None:
            nav = nav.loc[nav.index >= lower]
        if cutoff is not None:
            nav = nav.loc[nav.index <= cutoff]
    missing_count = int(nav.isna().sum())
    nav = nav.dropna()
    if not np.isfinite(nav.to_numpy()).all():
        raise ValueError("NAV values must be finite; infinity is not a valid missing value")
    if (nav <= 0).any():
        raise ValueError("NAV values must be strictly positive")
    if nav.empty:
        raise ValueError("At least one valid NAV observation is required")
    return nav.rename("nav_eur"), missing_count


def calculate_metrics(
    nav: pd.Series,
    *,
    transactions: pd.DataFrame | None = None,
    periods_per_year: int = 252,
    start_date: date | datetime | str | None = None,
    end_date: date | datetime | str | None = None,
    as_of: date | datetime | str | None = None,
) -> dict[str, Any]:
    """Calculate standard return, risk, win-rate, turnover, and fee metrics.

    Risk-free return is fixed to zero. Volatility and Sharpe use sample standard
    deviation (ddof=1) over observation-to-observation returns and sqrt(periods
    per year). Sortino uses the root mean square of negative returns, including
    zero for non-negative periods, as downside deviation.
    """

    if (
        isinstance(periods_per_year, bool)
        or not isinstance(periods_per_year, int)
        or periods_per_year <= 0
    ):
        raise ValueError("periods_per_year must be a positive integer")
    values, missing_count = validate_nav(
        nav,
        start_date=start_date,
        end_date=end_date,
        as_of=as_of,
    )
    first_nav = float(values.iloc[0])
    final_nav = float(values.iloc[-1])
    total_return = final_nav / first_nav - 1.0
    if not math.isfinite(total_return):
        raise ValueError("Computed total return is not finite")
    years = _calendar_year_fraction(values.index[0], values.index[-1])
    cagr = (final_nav / first_nav) ** (1.0 / years) - 1.0 if years > 0 else None
    returns = observation_returns(values)
    if not np.isfinite(returns.to_numpy()).all():
        raise ValueError("Computed observation returns are not finite")
    volatility = (
        float(returns.std(ddof=1) * math.sqrt(periods_per_year))
        if len(returns) >= 2
        else None
    )
    return_std = float(returns.std(ddof=1)) if len(returns) >= 2 else None
    sharpe = (
        float(returns.mean() / return_std * math.sqrt(periods_per_year))
        if return_std is not None and return_std > 0
        else None
    )
    downside_deviation = (
        float(np.sqrt(np.square(np.minimum(returns.to_numpy(), 0.0)).mean()))
        if not returns.empty
        else None
    )
    sortino = (
        float(returns.mean() / downside_deviation * math.sqrt(periods_per_year))
        if downside_deviation is not None and downside_deviation > 0
        else None
    )
    max_drawdown = maximum_drawdown(values)
    calmar = (
        float(cagr / abs(max_drawdown))
        if cagr is not None and max_drawdown < 0
        else None
    )
    annual = calendar_period_returns(values, "year")
    monthly = calendar_period_returns(values, "month")
    best_year = float(annual["return"].max()) if not annual.empty else None
    worst_year = float(annual["return"].min()) if not annual.empty else None
    best_month = float(monthly["return"].max()) if not monthly.empty else None
    worst_month = float(monthly["return"].min()) if not monthly.empty else None

    transaction_frame = transactions.copy() if transactions is not None else pd.DataFrame()
    if not transaction_frame.empty:
        required = {"execution_date", "notional_eur", "fee_eur"}
        missing_columns = required - set(transaction_frame.columns)
        if missing_columns:
            raise ValueError(f"transactions missing columns: {sorted(missing_columns)}")
        execution_dates = pd.to_datetime(transaction_frame["execution_date"], errors="raise")
        value_dates = pd.DatetimeIndex(values.index)
        if value_dates.tz is not None and execution_dates.dt.tz is None:
            execution_dates = execution_dates.dt.tz_localize(value_dates.tz)
        elif value_dates.tz is None and execution_dates.dt.tz is not None:
            execution_dates = execution_dates.dt.tz_localize(None)
        lower = value_dates[0]
        upper = value_dates[-1]
        transaction_frame = transaction_frame.loc[
            (execution_dates >= lower) & (execution_dates <= upper)
        ].copy()
        transaction_frame["notional_eur"] = pd.to_numeric(
            transaction_frame["notional_eur"], errors="raise"
        )
        transaction_frame["fee_eur"] = pd.to_numeric(
            transaction_frame["fee_eur"], errors="raise"
        )
        amounts = transaction_frame[["notional_eur", "fee_eur"]].to_numpy()
        if not np.isfinite(amounts).all() or (amounts < 0).any():
            raise ValueError("Transaction notionals and fees must be finite and non-negative")
        total_fees = float(transaction_frame["fee_eur"].sum())
        traded_notional = float(transaction_frame["notional_eur"].sum())
    else:
        total_fees = 0.0
        traded_notional = 0.0
    if not math.isfinite(total_fees) or not math.isfinite(traded_notional):
        raise ValueError("Aggregated transaction fees and notionals must be finite")
    average_nav = float(values.mean())
    turnover = traded_notional / average_nav if average_nav > 0 else None

    return {
        "observation_count": len(values),
        "missing_nav_observations_dropped": missing_count,
        "start_date": values.index[0].date().isoformat(),
        "end_date": values.index[-1].date().isoformat(),
        "calendar_years": years,
        "initial_nav_eur": first_nav,
        "final_nav_eur": final_nav,
        "total_return": total_return,
        "cagr": cagr,
        "maximum_drawdown": max_drawdown,
        "annualized_volatility": volatility,
        "sharpe_ratio": sharpe,
        "sortino_ratio": sortino,
        "calmar_ratio": calmar,
        "best_year_return": best_year,
        "worst_year_return": worst_year,
        "best_month_return": best_month,
        "worst_month_return": worst_month,
        "win_rate": float((returns > 0).mean()) if not returns.empty else None,
        "observation_return_count": len(returns),
        "periods_per_year": periods_per_year,
        "risk_free_rate": 0.0,
        "total_traded_notional_eur": traded_notional,
        "turnover": turnover,
        "total_fees_eur": total_fees,
        "defined_drawdown_present": bool(max_drawdown < 0),
    }


def analyze_performance(
    nav: pd.Series,
    *,
    transactions: pd.DataFrame | None = None,
    periods_per_year: int = 252,
    start_date: date | datetime | str | None = None,
    end_date: date | datetime | str | None = None,
    as_of: date | datetime | str | None = None,
) -> dict[str, Any]:
    """Return metrics plus period-return and risk tables for reporting and plots."""

    values, missing_count = validate_nav(
        nav,
        start_date=start_date,
        end_date=end_date,
        as_of=as_of,
    )
    metrics = calculate_metrics(
        values,
        transactions=transactions,
        periods_per_year=periods_per_year,
    )
    metrics["missing_nav_observations_dropped"] = missing_count
    return {
        "metrics": metrics,
        "nav": values,
        "returns": observation_returns(values),
        "drawdowns": drawdown_series(values),
        "annual_returns": calendar_period_returns(values, "year"),
        "monthly_returns": calendar_period_returns(values, "month"),
    }


__all__ = [
    "analyze_performance",
    "calculate_metrics",
    "maximum_drawdown",
    "validate_nav",
]
