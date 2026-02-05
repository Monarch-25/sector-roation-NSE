# Optimization Scripts

This folder contains Optuna-based hyperparameter optimization scripts.

## Available Scripts

| Script | Strategy | Description |
|--------|----------|-------------|
| `optimize_rrg.py` | RRG | Optimize RS-Ratio, momentum, volatility params |
| `optimize_volume_collapse.py` | Volume Collapse | Optimize RRG + geometric volume params |

## Usage

Run from the project root directory:

```bash
# RRG Strategy (with 2-year holdout for true OOS)
python optimization/optimize_rrg.py --n-trials 100 --holdout-years 2

# Volume Collapse Strategy  
python optimization/optimize_volume_collapse.py --n-trials 100 --holdout-years 2
```

## Options

| Argument | Default | Description |
|----------|---------|-------------|
| `--csv` | `data/processed/indian_sector_data_2013_2025.csv` | Data file |
| `--n-trials` | 100 | Number of Optuna trials |
| `--objective` | `sharpe_ratio` | Metric to optimize |
| `--train-window` | 504 | Training window (days) |
| `--test-window` | 63 | Test window (days) |
| `--holdout-years` | 2.0 | **Years to reserve for true OOS testing** |
| `--output-dir` | `optimization` | Output directory |

## Data Leakage Prevention

The scripts use a **holdout approach** to prevent meta-level overfitting:

1. **Optimization period**: Walk-forward validation runs ONLY on data before the holdout cutoff
2. **Holdout period**: The last N years (default: 2) are reserved for TRUE out-of-sample testing
3. **No leakage**: Hyperparameters are selected without ever seeing holdout data

```
|-------- Optimization Period --------|---- Holdout (True OOS) ----|
         Walk-forward runs here              Final evaluation here
         (params selected on this)           (never seen during opt)
```

## Objectives

- `sharpe_ratio`
- `sortino_ratio`
- `calmar_ratio`
- `total_return`
- `cagr`
- `custom` (Sharpe - 0.5 × |MaxDD|)

## Output

Results are saved as CSV files in the `optimization/` folder.

