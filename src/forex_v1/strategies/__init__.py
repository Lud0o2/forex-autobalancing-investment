"""Portfolio allocation strategy interfaces and reference implementations."""

from forex_v1.strategies.benchmarks import (
    BENCHMARK_NAMES,
    BENCHMARK_REBALANCE_POLICIES,
    BuyAndHoldEqualWeightProvider,
    EqualWeightWeeklyProvider,
    EurCashProvider,
    InverseVolatility60Provider,
    Momentum60Provider,
    create_benchmark_provider,
    required_features_for_benchmark,
)
from forex_v1.strategies.fixed_weight import DEFAULT_FIXED_WEIGHTS, FixedWeightProvider

__all__ = [
    "BENCHMARK_NAMES",
    "BENCHMARK_REBALANCE_POLICIES",
    "DEFAULT_FIXED_WEIGHTS",
    "BuyAndHoldEqualWeightProvider",
    "EqualWeightWeeklyProvider",
    "EurCashProvider",
    "FixedWeightProvider",
    "InverseVolatility60Provider",
    "Momentum60Provider",
    "create_benchmark_provider",
    "required_features_for_benchmark",
]
