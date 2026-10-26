"""Risk and drawdown analysis for dated portfolio value series."""

from __future__ import annotations

import pandas as pd


def drawdown_series(nav: pd.Series) -> pd.Series:
    """Return the percentage decline from the running NAV peak."""

    return (nav / nav.cummax() - 1.0).rename("drawdown")


def maximum_drawdown(nav: pd.Series) -> float:
    """Return the most negative peak-to-trough drawdown, or zero if none exists."""

    if nav.empty:
        return 0.0
    return float(drawdown_series(nav).min())


def drawdown_episodes(nav: pd.Series, limit: int = 5) -> pd.DataFrame:
    """Return the largest underwater episodes with peak, trough, and recovery dates."""

    if limit < 1:
        raise ValueError("limit must be positive")
    if nav.empty:
        return pd.DataFrame(
            columns=["peak_date", "trough_date", "recovery_date", "drawdown"]
        )

    drawdowns = drawdown_series(nav)
    episodes: list[tuple[float, dict[str, object]]] = []
    underwater = drawdowns < 0
    i = 0
    while i < len(nav):
        if not underwater.iloc[i]:
            i += 1
            continue
        start = i
        while i < len(nav) and underwater.iloc[i]:
            i += 1
        end = i - 1
        segment = drawdowns.iloc[start : end + 1]
        trough_position = start + int(segment.argmin())
        prior_peak_start = int(nav.iloc[:start].argmax())
        peak_value = float(nav.iloc[prior_peak_start])
        recovery_date = None
        recovered = nav.iloc[end + 1 :][nav.iloc[end + 1 :] >= peak_value]
        if not recovered.empty:
            recovery_date = recovered.index[0]
        episode_drawdown = float(drawdowns.iloc[trough_position])
        episodes.append(
            (
                episode_drawdown,
                {
                "peak_date": nav.index[prior_peak_start],
                "trough_date": nav.index[trough_position],
                "recovery_date": recovery_date,
                "drawdown": episode_drawdown,
                },
            )
        )
    episodes.sort(key=lambda episode: episode[0])
    return pd.DataFrame([episode for _, episode in episodes[:limit]])
