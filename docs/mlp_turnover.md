# MLP V1 turnover regularization (step 8D)

## Objective

The one step 8D experiment retains the existing 91 → 32 → 16 → 8 MLP, AdamW,
learning rate, seed, chronological folds, fees, differentiable growth loss, and
step 8C nested validation procedure. The only training objective change is the
fixed penalty coefficient `lambda_turnover = 0.001`, specified in
`configs/mlp_v1_turnover.yaml`. No search over this coefficient or other
hyperparameters is performed.

For each period, the model target weights are `w_t`. The pre-trade holdings
`q_t` are EUR-only at the first investment; afterward, they are the preceding
target holdings drifted by the preceding currency returns and renormalized by
the resulting portfolio value. Foreign turnover is

`u_t = sum_{i != EUR} abs(w_t[i] - q_t[i])`.

The objective is:

`loss_total = -mean(log(net_growth_factor_t)) + lambda_turnover * mean(u_t)`.

The penalty is calculated on the same drifted pre-trade weights used by the
existing differentiable fee proxy. It is differentiable through the target
allocations and previous-period drift. It counts both the initial investment
from 100% EUR and subsequent foreign-currency rotations.

`lambda_turnover` is a training regularizer only. It does not alter the exact
portfolio ledger or increase the configured 10 bps fee. All financial metrics
are still calculated from the existing backtester and actual transaction
journal.

## Experimental protocol

Before training the penalized MLP, the existing 8C model checkpoints are
replayed with the exact backtester at 10 bps and at zero fees. Equal Weight
Weekly is replayed under both fee settings as a reference. The zero-fee return
is the simulated gross allocation return; the difference between zero-fee and
10-bps NAV returns includes compounding and is not equal to the sum of fee EUR.

The A/B comparison uses:

- A: immutable step 8C checkpoint predictions, no retraining.
- B: the same fixed architecture, AdamW settings, learning rate, base seed,
  features, folds, nested inner validation and purge rules, with only the
  turnover regularizer enabled.
- Equal Weight Weekly and EUR Cash references.

Each fold's strategies use a common NAV calendar, capital, prices, Friday
signals, one conservative seven-calendar-day information lag, first market
observation after the signal for execution, and exact 10 bps ledger fees.
Turnover, transaction count, effective rebalances, fees, allocation weights,
and target-weight variation are reported per fold.

The run creates `reports/mlp_v1_turnover.md`,
`reports/mlp_v1_turnover.json`, comparative NAV/turnover charts, and immutable
checkpoints under `data/derived/models/mlp_v1_turnover/`. Run it with:

```powershell
.\.venv\Scripts\python.exe -m forex_v1.models.mlp_training --turnover --config configs\mlp_v1_turnover.yaml
```

The outer folds were examined during earlier steps 8B and 8C. Consequently,
these results are exploratory and cannot establish independent generalization
or be used to tune `lambda_turnover`. The final-test period from 2019-10-25
remains excluded. FRED publication availability is unknown and the data are not
certified executable prices.
