# Interpreting Volume Collapse Parameter Sensitivity Results

## What the notebook now tests
The sensitivity notebook uses two evaluation layers:

1. Optimization-period walk-forward (`wf_*` metrics)
- Trains on rolling historical windows.
- Tests on the next unseen window.
- Uses only pre-holdout data.

2. Holdout walk-forward (`holdout_*` metrics)
- Runs walk-forward on the full timeline.
- Keeps only out-of-sample predictions whose dates fall inside the holdout period.
- For each holdout prediction date, training uses only data available up to that point.

This avoids fitting once on the full holdout block and avoids static start-date weights.

## How to read the main outputs

### Top tables (`Top 20 by holdout_sharpe`, `Top 20 by wf_sharpe`)
- Prefer settings that are strong in both tables.
- If a parameter value ranks high in `wf_sharpe` but low in `holdout_sharpe`, treat it as possible overfit.
- If `holdout_total_return` improves but `holdout_max_drawdown` gets much worse, the improvement may be risk-inefficient.

### Response curves (per-parameter plots)
Each subplot shows metric response as one parameter changes and all others stay fixed.

- Flat curve: robust parameter (low sensitivity).
- Sharp peaks: fragile parameter; performance may depend on precise tuning.
- Wide plateau near the top: usually preferable for production.
- Large divergence between `WF Sharpe` and `Holdout Sharpe`: stability warning.

### Robustness table (`holdout_sharpe_span`)
- Higher span means the parameter has larger impact on holdout performance.
- Very high span can be useful (strong control knob) but also risky (high fragility).
- Compare `best_holdout_value` vs `base_value`:
  - Close values suggest the baseline is already near a robust region.
  - Far values suggest a candidate retune, but validate jointly with other parameters.

## Practical decision rules

1. Prioritize holdout metrics over optimization-period metrics.
2. Prefer parameter regions with stable holdout curves, not isolated spikes.
3. Change only a small number of high-impact parameters at once.
4. Re-run sensitivity after any major data or universe change.
5. Before deployment, run a final full walk-forward report and inspect drawdown profile, not Sharpe alone.
