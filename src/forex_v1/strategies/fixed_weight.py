"""Fixed long-only weights used as a configurable reference provider."""

from collections.abc import Mapping
from types import MappingProxyType

from forex_v1.contracts import TargetWeights
from forex_v1.portfolio.constraints import validate_target_weights

DEFAULT_FIXED_WEIGHTS: Mapping[str, float] = MappingProxyType(
    {
        "EUR": 0.25,
        "USD": 0.25,
        "GBP": 0.25,
        "JPY": 0.25,
    }
)


class FixedWeightProvider:
    """Return a validated fixed allocation; this is not an optimized strategy."""

    def __init__(
        self,
        target_weights: Mapping[str, float] = DEFAULT_FIXED_WEIGHTS,
    ) -> None:
        self.target_weights = validate_target_weights(target_weights)

    def get_target_weights(
        self,
        history_until_t,
        current_state,
    ) -> TargetWeights:
        return dict(self.target_weights)


__all__ = ["DEFAULT_FIXED_WEIGHTS", "FixedWeightProvider"]
