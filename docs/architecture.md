# Architecture

## V1 scope

Forex V1 is a local research project, not a trading system. Its assumptions are:

- EUR is the reference and reporting currency.
- The portfolio is long-only: target allocations and positions are non-negative.
- There is no borrowing or leverage.
- Market observations are daily (D1).
- The planned rebalance cadence is weekly.
- No broker, live order execution, web interface, or neural network is part of this
  architecture step.

The market-data schema is specified in `docs/data_contract.md`. Other financial rules will
be defined in their dedicated steps. This document defines module responsibilities and
boundaries, not their algorithms.

## Modules

| Module | Responsibility | Planned input | Planned output |
| --- | --- | --- | --- |
| `data` | Acquire, validate, and store source and normalized market data. | Source records or local data files. | Validated, dated market observations with their provenance. |
| `features` | Define transformations of historical observations into model/strategy inputs. | Historical market observations. | Dated feature data aligned to the input observations. |
| `strategies` | Provide non-ML allocation decisions through the shared target-weight contract. | History available at the signal time and caller-supplied current state. | `target_weights`, keyed by currency code. |
| `models` | Host future interchangeable predictive/allocation models. | The same as-of history and state contract used by strategies. | `target_weights`, keyed by currency code. |
| `portfolio` | Own portfolio-state, position, capital, and constraint interfaces. | Current portfolio state, target weights, and dated valuation data. | Portfolio state and valuation/trade result interfaces. |
| `backtest` | Coordinate historical simulation using dated inputs and a weight provider. | Market observations, a `TargetWeightProvider`, and simulation settings. | Dated simulation results, including a NAV series and any result records. |
| `evaluation` | Describe performance metrics and comparisons over completed results. | Dated simulation results. | Metric and comparison results. |
| `experiments` | Configure and coordinate experiment runs and comparisons. | Experiment settings and data references. | Run metadata and links/records for resulting simulations and metrics. |

These are planned boundaries only. No acquisition, feature, allocation, portfolio, simulation,
or metric algorithm is implemented by this architecture skeleton.

## Data flow

The intended processing flow is:

```text
source/local data
    -> data validation and storage
    -> historical market observations
    -> features (when needed)
    -> strategies or models
    -> target_weights
    -> portfolio and backtest
    -> dated simulation results
    -> evaluation
    -> experiment comparison
```

The signal-producing component must receive history only through its signal time. The exact
rules for signal dates, execution/valuation dates, and missing observations belong to later
data and simulation specifications; this skeleton does not invent them.

## Import dependencies

The flow above is a data-flow diagram. Python imports are directed from consumers toward
the interfaces they use; lower-level modules must not import orchestration or reporting
modules.

Allowed dependencies:

- `features` may use `data` interfaces.
- `strategies` and `models` may use `data` and `features` interfaces, and the shared
  contract in `forex_v1.contracts`.
- `portfolio` may use data and shared value contracts; it must not depend on a particular
  strategy or model implementation.
- `backtest` may use data and portfolio interfaces and the shared provider contract. It
  must not import a concrete class from `models` or `strategies`.
- `evaluation` may consume backtest result interfaces; backtest must not import evaluation.
- `experiments` may coordinate the other layers. No other layer may depend on
  `experiments`.
- `forex_v1.contracts` is a leaf shared interface module and must not import any of the
  feature, strategy, model, portfolio, backtest, evaluation, or experiment packages.

This keeps dependencies acyclic and prevents orchestration from becoming a dependency of
the components it coordinates.

## Shared target-weight contract

`forex_v1.contracts.TargetWeightProvider` defines the common interface for strategy and
model implementations:

```python
get_target_weights(
    history_until_t: pandas.DataFrame,
    current_state: Mapping[str, object],
) -> Mapping[str, float]
```

The returned mapping is called `target_weights` and is keyed by currency code. It expresses
the requested portfolio allocation, not orders, units, or realized holdings. In V1, its
intended use is long-only and unleveraged; detailed validation, normalization tolerances,
availability rules, and execution timing must be established in the later portfolio and
backtest steps. This interface does not supply defaults for missing or invalid values.

The protocol is structural: a strategy or model can satisfy it without inheriting from a
concrete base class. The backtest can therefore receive any provider that implements this
method and remains independent of the model's class and implementation.

## Replacing a future model

1. Implement `get_target_weights` on a new provider using only data available as of the
   signal time.
2. Return currency-keyed `target_weights` under the shared contract.
3. Pass that provider to the simulation orchestration through the contract; do not add a
   model-specific branch to `backtest`.
4. Keep training, preprocessing, and model-specific configuration inside the model or
   experiment layer rather than the portfolio engine.

The contract does not prescribe a neural-network framework, training procedure, feature
schema, or financial allocation algorithm.
