"""Explicit proportional conversion costs."""

import math
from numbers import Real


def validate_fee_rate(fee_rate: float) -> float:
    """Validate a fee fraction in [0, 1); e.g. 10 bps is 0.001."""
    if isinstance(fee_rate, bool) or not isinstance(fee_rate, Real):
        raise TypeError("fee_rate must be an explicitly configured numeric fraction")
    rate = float(fee_rate)
    if not math.isfinite(rate) or rate < 0 or rate >= 1:
        raise ValueError("fee_rate must be finite and in [0, 1)")
    return rate


def fee_rate_from_bps(basis_points: float) -> float:
    """Convert basis points to a fraction without interpreting percentage input."""
    if isinstance(basis_points, bool) or not isinstance(basis_points, Real):
        raise TypeError("basis_points must be numeric")
    bps = float(basis_points)
    if not math.isfinite(bps) or bps < 0:
        raise ValueError("basis_points must be finite and non-negative")
    return validate_fee_rate(bps / 10_000.0)
