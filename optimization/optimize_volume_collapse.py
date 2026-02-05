"""
Hyperparameter Optimization for Volume Collapse Strategy

Uses Optuna (Bayesian optimization) to find optimal parameters
based on walk-forward validation performance.

IMPORTANT: Uses a holdout period to prevent data leakage.
- Optimization runs on data BEFORE the holdout cutoff
- Final evaluation runs on the holdout period (true OOS)

Usage:
    python optimization/optimize_volume_collapse.py --n-trials 100
    python optimization/optimize_volume_collapse.py --n-trials 50 --holdout-years 2
"""
import argparse
import sys
import warnings
from datetime import datetime
from pathlib import Path

import optuna
from optuna.samplers import TPESampler
import pandas as pd
import numpy as np

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from strategies import VolumeCollapseStrategy, VolumeCollapseConfig
from backtesting import Backtester, WalkForwardValidator, WalkForwardConfig

warnings.filterwarnings('ignore')


def load_data(csv_path: str) -> tuple:
    """Load price data and extract benchmark."""
    df = pd.read_csv(csv_path, parse_dates=["Date"])
    df = df.set_index("Date").sort_index()
    df = df.apply(pd.to_numeric, errors="coerce")
    df = df.ffill().dropna(how="all")
    
    if "Benchmark" in df.columns:
        benchmark = df["Benchmark"]
        prices = df.drop(columns=["Benchmark"])
    else:
        prices = df
        returns = prices.pct_change().fillna(0)
        benchmark = (1 + returns.mean(axis=1)).cumprod()
        benchmark.name = "Benchmark"
    
    return prices, benchmark


def create_objective(prices, benchmark, wf_config, objective_metric="sharpe_ratio"):
    """Create Optuna objective function."""
    
    def objective(trial):
        # Sample hyperparameters
        config = VolumeCollapseConfig(
            # RRG parameters
            top_n_sectors=trial.suggest_int("top_n_sectors", 2, 6),
            max_sector_weight=trial.suggest_float("max_sector_weight", 0.25, 0.50, step=0.05),
            min_sector_weight=trial.suggest_float("min_sector_weight", 0.01, 0.05, step=0.01),
            rs_lookback=trial.suggest_int("rs_lookback", 20, 100, step=10),
            momentum_lookback=trial.suggest_int("momentum_lookback", 5, 30, step=5),
            volatility_window=trial.suggest_int("volatility_window", 10, 52, step=6),
            use_trend_filter=trial.suggest_categorical("use_trend_filter", [True, False]),
            trend_ma_period=trial.suggest_int("trend_ma_period", 20, 60, step=10),
            weight_smoothing_alpha=trial.suggest_float("weight_smoothing_alpha", 0.1, 0.5, step=0.1),
            full_allocation=True,
            rebalance_frequency="W-FRI",
            
            # Volume Collapse parameters
            vol_window=trial.suggest_int("vol_window", 30, 100, step=10),
            vol_percentile=trial.suggest_float("vol_percentile", 0.05, 0.30, step=0.05),
            risk_reduction_factor=trial.suggest_float("risk_reduction_factor", 0.3, 0.7, step=0.1),
            min_exposure=trial.suggest_float("min_exposure", 0.1, 0.4, step=0.1),
            smooth_scaling=trial.suggest_categorical("smooth_scaling", [True, False]),
        )
        
        try:
            # Run walk-forward validation
            strategy = VolumeCollapseStrategy(config)
            validator = WalkForwardValidator(strategy, config=wf_config)
            result = validator.validate(prices, benchmark)
            
            # Get objective metric
            metrics = result.aggregate_metrics
            
            if objective_metric == "sharpe_ratio":
                score = metrics.get("sharpe_ratio", -999)
            elif objective_metric == "sortino_ratio":
                score = metrics.get("sortino_ratio", -999)
            elif objective_metric == "calmar_ratio":
                score = metrics.get("calmar_ratio", -999)
            elif objective_metric == "total_return":
                score = metrics.get("total_return", -999)
            elif objective_metric == "cagr":
                score = metrics.get("cagr", -999)
            else:
                # Custom: Sharpe - 0.5 * abs(max_drawdown)
                sharpe = metrics.get("sharpe_ratio", 0)
                mdd = abs(metrics.get("max_drawdown", -1))
                score = sharpe - 0.5 * mdd
            
            # Handle NaN/Inf
            if np.isnan(score) or np.isinf(score):
                return -999
            
            return score
            
        except Exception as e:
            print(f"Trial failed: {e}")
            return -999
    
    return objective


def run_optimization(
    csv_path: str,
    n_trials: int = 100,
    objective_metric: str = "sharpe_ratio",
    train_window: int = 504,
    test_window: int = 63,
    holdout_years: float = 2.0,
    output_dir: str = "."
):
    """Run hyperparameter optimization with holdout period."""
    
    print("=" * 70)
    print("VOLUME COLLAPSE STRATEGY HYPERPARAMETER OPTIMIZATION")
    print("=" * 70)
    
    # Load full data
    print("\nLoading data...")
    prices_full, benchmark_full = load_data(csv_path)
    print(f"Full data: {prices_full.index[0].date()} to {prices_full.index[-1].date()}")
    print(f"Total: {len(prices_full)} days, {len(prices_full.columns)} sectors")
    
    # Split into optimization and holdout periods
    holdout_days = int(holdout_years * 252)  # Trading days per year
    cutoff_idx = len(prices_full) - holdout_days
    cutoff_date = prices_full.index[cutoff_idx]
    
    prices_opt = prices_full.iloc[:cutoff_idx]
    benchmark_opt = benchmark_full.iloc[:cutoff_idx]
    
    prices_holdout = prices_full.iloc[cutoff_idx:]
    benchmark_holdout = benchmark_full.iloc[cutoff_idx:]
    
    print(f"\n--- Data Split ---")
    print(f"Optimization period: {prices_opt.index[0].date()} to {prices_opt.index[-1].date()} ({len(prices_opt)} days)")
    print(f"Holdout period:      {prices_holdout.index[0].date()} to {prices_holdout.index[-1].date()} ({len(prices_holdout)} days)")
    print(f"Holdout years:       {holdout_years}")
    
    print(f"\n--- Optimization Settings ---")
    print(f"Objective: Maximize {objective_metric}")
    print(f"Trials: {n_trials}")
    print(f"Walk-Forward: train={train_window} days, test={test_window} days")
    
    # Walk-forward config (runs on optimization period only)
    wf_config = WalkForwardConfig(
        train_window=train_window,
        test_window=test_window,
        step_size=test_window
    )
    
    # Create study
    sampler = TPESampler(seed=42)
    study = optuna.create_study(
        direction="maximize",
        sampler=sampler,
        study_name="volume_collapse_optimization"
    )
    
    # Create objective using ONLY optimization period data
    objective = create_objective(prices_opt, benchmark_opt, wf_config, objective_metric)
    
    # Run optimization
    print(f"\nStarting optimization with {n_trials} trials...")
    print("-" * 70)
    
    study.optimize(
        objective,
        n_trials=n_trials,
        show_progress_bar=True,
        n_jobs=1
    )
    
    # Results
    print("\n" + "=" * 70)
    print("OPTIMIZATION RESULTS (on optimization period)")
    print("=" * 70)
    
    print(f"\nBest {objective_metric}: {study.best_value:.4f}")
    print("\nBest Parameters:")
    for param, value in study.best_params.items():
        print(f"  {param}: {value}")
    
    # Create best config
    best_config = VolumeCollapseConfig(
        top_n_sectors=study.best_params["top_n_sectors"],
        max_sector_weight=study.best_params["max_sector_weight"],
        min_sector_weight=study.best_params["min_sector_weight"],
        rs_lookback=study.best_params["rs_lookback"],
        momentum_lookback=study.best_params["momentum_lookback"],
        volatility_window=study.best_params["volatility_window"],
        use_trend_filter=study.best_params["use_trend_filter"],
        trend_ma_period=study.best_params["trend_ma_period"],
        weight_smoothing_alpha=study.best_params["weight_smoothing_alpha"],
        full_allocation=True,
        rebalance_frequency="W-FRI",
        vol_window=study.best_params["vol_window"],
        vol_percentile=study.best_params["vol_percentile"],
        risk_reduction_factor=study.best_params["risk_reduction_factor"],
        min_exposure=study.best_params["min_exposure"],
        smooth_scaling=study.best_params["smooth_scaling"],
    )
    
    # =========================================================================
    # TRUE OUT-OF-SAMPLE EVALUATION ON HOLDOUT PERIOD
    # =========================================================================
    print("\n" + "=" * 70)
    print("TRUE OUT-OF-SAMPLE EVALUATION (Holdout Period)")
    print("=" * 70)
    print(f"Period: {prices_holdout.index[0].date()} to {prices_holdout.index[-1].date()}")
    print("Note: These params were NEVER optimized on this data!\n")
    
    # Fit on optimization period, test on holdout
    strategy = VolumeCollapseStrategy(best_config)
    strategy.fit(prices_opt, benchmark_opt)
    
    # Run backtest on holdout period
    backtester = Backtester(strategy, risk_free_rate=0.05)
    holdout_result = backtester.run(prices_holdout, benchmark_holdout)
    
    print("--- Holdout Period Performance ---")
    backtester.print_report(holdout_result)
    
    # Also run walk-forward on full data for comparison
    print("\n" + "-" * 70)
    print("Walk-Forward on FULL data (for reference, includes optimization period)")
    
    strategy_full = VolumeCollapseStrategy(best_config)
    validator_full = WalkForwardValidator(strategy_full, config=wf_config, risk_free_rate=0.05)
    wf_result_full = validator_full.validate(prices_full, benchmark_full)
    validator_full.print_report(wf_result_full)
    
    # Save results
    results_df = study.trials_dataframe()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_file = Path(output_dir) / f"volume_collapse_optimization_{timestamp}.csv"
    results_df.to_csv(results_file, index=False)
    print(f"\nSaved trial results to {results_file}")
    
    # Print config for copy-paste
    print("\n" + "=" * 70)
    print("COPY-PASTE CONFIG:")
    print("=" * 70)
    print(f"""
config = VolumeCollapseConfig(
    # RRG parameters
    top_n_sectors={study.best_params["top_n_sectors"]},
    max_sector_weight={study.best_params["max_sector_weight"]},
    min_sector_weight={study.best_params["min_sector_weight"]},
    rs_lookback={study.best_params["rs_lookback"]},
    momentum_lookback={study.best_params["momentum_lookback"]},
    volatility_window={study.best_params["volatility_window"]},
    use_trend_filter={study.best_params["use_trend_filter"]},
    trend_ma_period={study.best_params["trend_ma_period"]},
    weight_smoothing_alpha={study.best_params["weight_smoothing_alpha"]},
    full_allocation=True,
    rebalance_frequency="W-FRI",
    # Volume Collapse parameters
    vol_window={study.best_params["vol_window"]},
    vol_percentile={study.best_params["vol_percentile"]},
    risk_reduction_factor={study.best_params["risk_reduction_factor"]},
    min_exposure={study.best_params["min_exposure"]},
    smooth_scaling={study.best_params["smooth_scaling"]},
)
""")
    
    return study, best_config, holdout_result


def main():
    parser = argparse.ArgumentParser(description="Optimize Volume Collapse strategy hyperparameters")
    
    parser.add_argument("--csv", default="data/processed/indian_sector_data_2013_2025.csv")
    parser.add_argument("--n-trials", type=int, default=100, help="Number of optimization trials")
    parser.add_argument("--objective", default="sharpe_ratio", 
                        choices=["sharpe_ratio", "sortino_ratio", "calmar_ratio", "total_return", "cagr", "custom"],
                        help="Optimization objective")
    parser.add_argument("--train-window", type=int, default=504, help="Training window (days)")
    parser.add_argument("--test-window", type=int, default=63, help="Test window (days)")
    parser.add_argument("--holdout-years", type=float, default=2.0, 
                        help="Years to hold out for true OOS testing (default: 2)")
    parser.add_argument("--output-dir", default="optimization", help="Output directory for results")
    
    args = parser.parse_args()
    
    run_optimization(
        csv_path=args.csv,
        n_trials=args.n_trials,
        objective_metric=args.objective,
        train_window=args.train_window,
        test_window=args.test_window,
        holdout_years=args.holdout_years,
        output_dir=args.output_dir
    )


if __name__ == "__main__":
    main()
