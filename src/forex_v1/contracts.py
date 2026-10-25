"""Shared interfaces between interchangeable strategy and model implementations."""

from collections.abc import Mapping
from typing import Protocol

import pandas as pd

TargetWeights = Mapping[str, float]


class TargetWeightProvider(Protocol):
    """Produce target portfolio weights using information available at the signal time."""

    def get_target_weights(
        self,
        history_until_t: pd.DataFrame,
        current_state: Mapping[str, object],
    ) -> TargetWeights:
        """Return target weights keyed by currency code."""
