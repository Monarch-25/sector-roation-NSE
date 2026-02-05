# Hyperparameter Optimization Guide

This document explains the Bayesian optimization process for tuning RRG strategy parameters.

---

## Optimization Overview

The optimizer uses **Optuna's TPE (Tree-structured Parzen Estimator)** sampler for Bayesian optimization. Unlike grid search, TPE builds a probabilistic model to focus on promising parameter regions.

```
┌─────────────────────────────────────────────────────────────┐
│                    OPTIMIZATION LOOP                        │
├─────────────────────────────────────────────────────────────┤
│  1. Sample parameters from TPE prior                        │
│  2. Create RRGStrategy with sampled config                  │
│  3. Run walk-forward validation                             │
│  4. Record objective (Sharpe ratio)                         │
│  5. Update TPE posterior                                    │
│  6. Repeat for N trials                                     │
└─────────────────────────────────────────────────────────────┘
```

---

## Parameter Search Space

| Parameter | Range | Step | Description |
|-----------|-------|------|-------------|
| `top_n_sectors` | 2 – 6 | 1 | Number of sectors to hold |
| `max_sector_weight` | 0.25 – 0.50 | 0.05 | Maximum weight per sector |
| `min_sector_weight` | 0.01 – 0.05 | 0.01 | Minimum weight threshold |
| `rs_lookback` | 20 – 100 | 10 | RS-Ratio normalization window (weeks) |
| `momentum_lookback` | 5 – 30 | 5 | RS-Momentum calculation window |
| `volatility_window` | 10 – 52 | 6 | Inverse-vol weighting window |
| `use_trend_filter` | True/False | — | Enable benchmark trend filter |
| `trend_ma_period` | 20 – 60 | 10 | Moving average period for trend filter |
| `weight_smoothing_alpha` | 0.1 – 0.5 | 0.1 | EMA alpha for turnover reduction |

---

## Objective Functions

| Objective | Formula | Use Case |
|-----------|---------|----------|
| `sharpe_ratio` | (Return - Rf) / σ | **Default** – balanced risk-adjusted returns |
| `sortino_ratio` | (Return - Rf) / σ_downside | Penalizes downside risk more |
| `calmar_ratio` | CAGR / Max DD | Focuses on drawdown control |
| `total_return` | Cumulative return | Maximize raw returns |
| `cagr` | Annualized return | Time-normalized returns |
| `custom` | Sharpe - 0.5*|MaxDD| | Blend of Sharpe and drawdown |

---

## Optimal Configuration (Found)

```python
config = RRGConfig(
    top_n_sectors=5,
    max_sector_weight=0.3,
    min_sector_weight=0.01,
    rs_lookback=100,
    momentum_lookback=30,
    volatility_window=46,
    use_trend_filter=True,
    trend_ma_period=50,
    weight_smoothing_alpha=0.3,
    full_allocation=True,
    rebalance_frequency="W-FRI"
)
```

### Performance (Optimized vs Default)

| Metric | Default | Optimized | Δ |
|--------|---------|-----------|---|
| Total Return | 384% | 455% | +71% |
| CAGR | 13.1% | 14.3% | +1.2% |
| Sharpe | 0.539 | 0.612 | +0.07 |
| WF OOS Sharpe | 0.521 | 0.650 | +0.13 |
| Max Drawdown | -38.6% | -36.0% | +2.6% |
| Info Ratio | 0.265 | 0.504 | +0.24 |
| Fold Profitability | 59.6% | 70.2% | +10.6% |

---

## Usage

```bash
# Run 50 trials optimizing for Sharpe
python optimize_hyperparams.py --n-trials 50 --objective sharpe_ratio

# Run 100 trials optimizing for Sortino
python optimize_hyperparams.py --n-trials 100 --objective sortino_ratio

# Custom walk-forward windows
python optimize_hyperparams.py --n-trials 50 --train-window 756 --test-window 126
```

### Options

| Flag | Default | Description |
|------|---------|-------------|
| `--csv` | `data/processed/...csv` | Path to price data |
| `--n-trials` | 100 | Number of optimization trials |
| `--objective` | `sharpe_ratio` | Metric to maximize |
| `--train-window` | 504 | Walk-forward training days (~2 years) |
| `--test-window` | 63 | Walk-forward test days (~3 months) |

---

## Interpretation Tips

1. **Convergence**: If best value stops improving after ~30 trials, the search is likely converged
2. **Overfitting**: Compare WF OOS Sharpe to full backtest Sharpe; gap >0.2 suggests overfitting
3. **Robustness**: Run optimization multiple times with different seeds; stable params are more reliable
4. **Parameter Sensitivity**: Check `optimization_results_*.csv` to see how sensitive the objective is to each parameter

---

## Output Files

| File | Contents |
|------|----------|
| `optimization_results_YYYYMMDD_HHMMSS.csv` | All trials with parameters and scores |

Load and analyze:
```python
import pandas as pd
df = pd.read_csv("optimization_results_*.csv")
print(df.sort_values("value", ascending=False).head(10))
```
