"""Observation and calendar-period return calculations."""

from __future__ import annotations

import pandas as pd


def observation_returns(nav: pd.Series) -> pd.Series:
    """Compute simple returns between consecutive valid NAV observations."""

    return nav.pct_change(fill_method=None).dropna().rename("return")


def calendar_period_returns(nav: pd.Series, frequency: str) -> pd.DataFrame:
    """Compute calendar-period close-to-close returns and flag truncated edges.

    Each period return uses the previous observed NAV as its baseline. For the
    first available period, the first NAV is its baseline. Only periods truncated
    by the dataset boundaries are flagged partial; weekends and holidays do not
    count as missing market observations.
    """

    if frequency not in {"year", "month"}:
        raise ValueError("frequency must be 'year' or 'month'")
    if nav.empty:
        return pd.DataFrame(columns=["period", "return", "start_date", "end_date", "partial"])

    date_index = pd.DatetimeIndex(nav.index)
    period_labels = date_index.to_period("Y" if frequency == "year" else "M")
    grouped: list[dict[str, object]] = []
    first_period = period_labels[0]
    last_period = period_labels[-1]

    for period in period_labels.unique():
        values = nav.loc[period_labels == period]
        start_date = values.index[0]
        end_date = values.index[-1]
        previous = nav.loc[nav.index < start_date]
        start_value = float(previous.iloc[-1]) if not previous.empty else float(values.iloc[0])
        period_return = float(values.iloc[-1]) / start_value - 1.0
        if frequency == "year":
            partial = period == first_period and (
                nav.index[0].month != 1 or nav.index[0].day != 1
            )
            partial = partial or (period == last_period and nav.index[-1].month != 12)
        else:
            partial = period == first_period and nav.index[0].day != 1
            partial = partial or (
                period == last_period
                and nav.index[-1].day != nav.index[-1].days_in_month
            )
        grouped.append(
            {
                "period": str(period),
                "return": period_return,
                "start_date": start_date,
                "end_date": end_date,
                "partial": bool(partial),
            }
        )
    return pd.DataFrame(grouped)


def monthly_return_matrix(monthly_returns: pd.DataFrame) -> pd.DataFrame:
    """Pivot monthly returns into year-by-month values for a heatmap."""

    if monthly_returns.empty:
        return pd.DataFrame(index=pd.Index([], name="year"), columns=range(1, 13))
    values = monthly_returns.copy()
    values["year"] = values["period"].str.slice(0, 4).astype(int)
    values["month"] = values["period"].str.slice(5, 7).astype(int)
    return values.pivot(index="year", columns="month", values="return").reindex(columns=range(1, 13))
