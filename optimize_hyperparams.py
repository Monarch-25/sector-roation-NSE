"""
Hyperparameter Optimization for RRG Strategy

Uses Optuna (Bayesian optimization) to find optimal parameters
based on walk-forward validation performance.

Usage:
    python optimize_hyperparams.py --n-trials 100
    python optimize_hyperparams.py --n-trials 50 --objective sharpe
"""
import argparse
import warnings
from datetime import datetime

import optuna
from optuna.samplers import TPESampler
import pandas as pd
import numpy as np

from strategies import RRGStrategy, RRGConfig
from backtesting import WalkForwardValidator, WalkForwardConfig

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
        config = RRGConfig(
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
            rebalance_frequency="W-FRI"
        )
        
        try:
            # Run walk-forward validation
            strategy = RRGStrategy(config)
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
    test_window: int = 63
):
    """Run hyperparameter optimization."""
    
    print("=" * 60)
    print("RRG STRATEGY HYPERPARAMETER OPTIMIZATION")
    print("=" * 60)
    print(f"Objective: Maximize {objective_metric}")
    print(f"Trials: {n_trials}")
    print(f"Walk-Forward: train={train_window} days, test={test_window} days")
    print()
    
    # Load data
    print("Loading data...")
    prices, benchmark = load_data(csv_path)
    print(f"Data: {len(prices)} days, {len(prices.columns)} sectors")
    
    # Walk-forward config
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
        study_name="rrg_optimization"
    )
    
    # Create objective
    objective = create_objective(prices, benchmark, wf_config, objective_metric)
    
    # Run optimization
    print(f"\nStarting optimization with {n_trials} trials...")
    print("-" * 60)
    
    study.optimize(
        objective,
        n_trials=n_trials,
        show_progress_bar=True,
        n_jobs=1  # Sequential for stability
    )
    
    # Results
    print("\n" + "=" * 60)
    print("OPTIMIZATION RESULTS")
    print("=" * 60)
    
    print(f"\nBest {objective_metric}: {study.best_value:.4f}")
    print("\nBest Parameters:")
    for param, value in study.best_params.items():
        print(f"  {param}: {value}")
    
    # Create best config
    best_config = RRGConfig(
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
        rebalance_frequency="W-FRI"
    )
    
    # Run final walk-forward with best params
    print("\n" + "-" * 60)
    print("Running final walk-forward with best parameters...")
    
    strategy = RRGStrategy(best_config)
    validator = WalkForwardValidator(strategy, config=wf_config)
    result = validator.validate(prices, benchmark)
    
    validator.print_report(result)
    
    # Save results
    results_df = study.trials_dataframe()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_file = f"optimization_results_{timestamp}.csv"
    results_df.to_csv(results_file, index=False)
    print(f"\nSaved trial results to {results_file}")
    
    # Print config for copy-paste
    print("\n" + "=" * 60)
    print("COPY-PASTE CONFIG:")
    print("=" * 60)
    print(f"""
config = RRGConfig(
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
    rebalance_frequency="W-FRI"
)
""")
    
    return study, best_config


def main():
    parser = argparse.ArgumentParser(description="Optimize RRG strategy hyperparameters")
    
    parser.add_argument("--csv", default="data/processed/indian_sector_data_2013_2025.csv")
    parser.add_argument("--n-trials", type=int, default=100, help="Number of optimization trials")
    parser.add_argument("--objective", default="sharpe_ratio", 
                        choices=["sharpe_ratio", "sortino_ratio", "calmar_ratio", "total_return", "cagr", "custom"],
                        help="Optimization objective")
    parser.add_argument("--train-window", type=int, default=504, help="Training window (days)")
    parser.add_argument("--test-window", type=int, default=63, help="Test window (days)")
    
    args = parser.parse_args()
    
    run_optimization(
        csv_path=args.csv,
        n_trials=args.n_trials,
        objective_metric=args.objective,
        train_window=args.train_window,
        test_window=args.test_window
    )


if __name__ == "__main__":
    main()
