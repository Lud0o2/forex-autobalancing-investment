# Weekly historical backtesting (step 6B)

## Scope

The backtest coordinates the dated EUR-converted rates in `data/derived/eur_rates/`
with any implementation of `TargetWeightProvider` and the 6A portfolio accounting
engine. The included `FixedWeightProvider` is a configurable test strategy, not an
optimized allocation or an AI model. The example scenario uses EUR/USD/GBP/JPY at
25% each, EUR 10,000 initial capital, and an explicit illustrative 10 bps fee.

The backtest has no leverage, short positions, interest, carry, broker integration,
or execution guarantees. FRED H.10 rates are indicative reference rates, not
certified executable prices. Spreads, slippage, settlement, and tax are not modeled.

## Calendar and information cutoff

Signals are generated on every calendar Friday in the simulation period, including
Fridays with no market observation. For a signal date `T`, the cutoff is
`T - availability_lag_days` calendar days (at least seven days in V1). The provider
receives price and feature history no later than the latest actual market date at
or before that cutoff. Its current-state prices are also marked using that eligible
price row. The source market date is preserved; it is never relabeled as the actual
publication or availability date. The provider history explicitly contains
`available_at=None` and `availability_known=False` for the FRED-derived prices.

FRED historical publication availability is unknown and not certified point-in-time.
The seven-calendar-day lag is a conservative modeling hypothesis only. It does not
prove when a specific observation became available.

## Execution and valuation chronology

The target allocation is created from the cutoff-limited history first. The
portfolio is then valued each day on every available common-calendar market
observation. A scheduled target is executed at the first observation strictly
after its Friday signal, using that observation's EUR rates and the portfolio
rebalance implementation from step 6A. Thus the execution rate does not flow back
into the earlier decision. Fridays without an observation still signal; holidays
and other gaps move execution to the next observed date. A final signal with no
later observation is recorded with `no_execution_observation` and does not create
an order. If sparse data make two Friday signals map to the same next observation,
the later overlapping signal is recorded as `execution_collision` rather than
double-executed.

The generic scenario setting `rebalance_policy` defaults to `weekly`. Its
`initial_only` value schedules the first valid target once and records later Friday
signals as `skipped_initial_only`; it is used for genuine Buy & Hold without making
the engine depend on that strategy's name. The selected policy is independent of
the provider's `TargetWeightProvider` implementation.

Daily NAV, EUR cash, currency quantities, values, and realized weights are recorded
after valuation and any order due on that observation. Transaction fees are taken
from the portfolio and separately enumerated by the portfolio layer. The annual
return, financing and interest of cash are not synthesized.

## Scenario and local command

The reproducible local scenario is in `configs/backtest.yaml`; it leaves the project
configuration's unconfigured cost untouched. Run it without network access:

```powershell
.\.venv\Scripts\python.exe -m forex_v1.backtest.reporting
```

Optional arguments:

```powershell
.\.venv\Scripts\python.exe -m forex_v1.backtest.reporting `
  --config configs\backtest.yaml `
  --eur-data data\derived\eur_rates `
  --feature-data data\derived\features `
  --reports-dir reports
```

The feature input is optional. When supplied, its provenance is checked against the
exact EUR Parquet file; a provider still receives only features dated within the
same cutoff-limited history.

## Outputs

The runner writes local, Git-ignored files:

- `reports/backtest_nav.csv`: NAV and EUR cash for each market observation.
- `reports/backtest_allocations.csv`: quantities, values, and realized weights
  across the eight currencies on each observation.
- `reports/backtest_decisions.csv`: Friday signal, cutoff, information market
  date, target and status.
- `reports/backtest_transactions.csv`: executed trade notionals, quantities and
  fees, separated by side and currency.
- `reports/backtest_capital_curve.png`: NAV in EUR.
- `reports/backtest_validation.md`: scenario, chronology, data provenance, first
  decision/execution dates and aggregate results.

The CSV decision and execution fields deliberately keep signal, information, and
execution dates separate. This output is a research backtest, not a record of
historically executable orders.
