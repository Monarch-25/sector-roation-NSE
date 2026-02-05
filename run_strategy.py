"""
Run Sector Rotation Strategy

CLI entry point for running backtests or walk-forward validation.

Usage:
    python run_strategy.py --mode backtest
    python run_strategy.py --mode walkforward
    python run_strategy.py --mode backtest --top-n 3 --max-weight 0.4
"""
import argparse
from pathlib import Path

import pandas as pd

from strategies import RRGStrategy, RRGConfig
from backtesting import Backtester, WalkForwardValidator, WalkForwardConfig


def load_data(csv_path: str) -> tuple:
    """Load price data and extract benchmark."""
    df = pd.read_csv(csv_path, parse_dates=["Date"])
    df = df.set_index("Date").sort_index()
    df = df.apply(pd.to_numeric, errors="coerce")
    df = df.ffill().dropna(how="all")
    
    # Separate benchmark and sector prices
    if "Benchmark" in df.columns:
        benchmark = df["Benchmark"]
        prices = df.drop(columns=["Benchmark"])
    else:
        # Build custom benchmark if not present
        prices = df
        returns = prices.pct_change().fillna(0)
        benchmark = (1 + returns.mean(axis=1)).cumprod()
        benchmark.name = "Benchmark"
    
    return prices, benchmark


def run_backtest(args) -> None:
    """Run full backtest."""
    print(f"Loading data from {args.csv}...")
    prices, benchmark = load_data(args.csv)
    
    print(f"Data range: {prices.index[0]} to {prices.index[-1]}")
    print(f"Sectors: {list(prices.columns)}")
    
    # Create strategy config
    config = RRGConfig(
        top_n_sectors=args.top_n,
        max_sector_weight=args.max_weight,
        min_sector_weight=args.min_weight,
        use_trend_filter=args.trend_filter,
        rs_lookback=args.rs_lookback,
        momentum_lookback=args.mom_lookback,
        volatility_window=args.vol_window
    )
    
    print(f"\nStrategy Config: {config}")
    
    # Create strategy and backtester
    strategy = RRGStrategy(config)
    backtester = Backtester(strategy, risk_free_rate=args.risk_free_rate)
    
    # Run backtest
    print("\nRunning backtest...")
    result = backtester.run(prices, benchmark)
    
    # Print report
    backtester.print_report(result)
    
    # Show latest weights
    print("\n--- Current Portfolio (Most Recent) ---")
    latest_date = result.weights.index[-1]
    latest_weights = result.weights.loc[latest_date].sort_values(ascending=False)
    latest_weights = latest_weights[latest_weights > 0]
    
    if len(latest_weights) == 0:
        print("No active positions (possibly in cash due to trend filter)")
    else:
        for sector, weight in latest_weights.items():
            print(f"  {sector}: {weight:.2%}")
    
    # Show RRG quadrant classification
    print("\n--- RRG Quadrant Classification ---")
    quadrants = strategy.get_quadrant_classification(latest_date)
    print(quadrants.sort_values("RS_Momentum", ascending=False).to_string())
    
    return result


def run_walkforward(args) -> None:
    """Run walk-forward validation."""
    print(f"Loading data from {args.csv}...")
    prices, benchmark = load_data(args.csv)
    
    print(f"Data range: {prices.index[0]} to {prices.index[-1]}")
    print(f"Sectors: {list(prices.columns)}")
    
    # Create strategy config
    strategy_config = RRGConfig(
        top_n_sectors=args.top_n,
        max_sector_weight=args.max_weight,
        min_sector_weight=args.min_weight,
        use_trend_filter=args.trend_filter
    )
    
    # Create walk-forward config
    wf_config = WalkForwardConfig(
        train_window=args.train_window,
        test_window=args.test_window,
        step_size=args.step_size
    )
    
    print(f"\nStrategy Config: {strategy_config}")
    print(f"Walk-Forward Config: {wf_config}")
    
    # Create strategy and validator
    strategy = RRGStrategy(strategy_config)
    validator = WalkForwardValidator(
        strategy, 
        config=wf_config,
        risk_free_rate=args.risk_free_rate
    )
    
    # Run validation
    print("\nRunning walk-forward validation...")
    result = validator.validate(prices, benchmark)
    
    # Print report
    validator.print_report(result)
    
    return result


def main():
    parser = argparse.ArgumentParser(
        description="Run sector rotation strategy backtest or validation"
    )
    
    # Mode selection
    parser.add_argument(
        "--mode", 
        choices=["backtest", "walkforward"],
        default="backtest",
        help="Run mode: backtest or walkforward"
    )
    
    # Data path
    parser.add_argument(
        "--csv",
        default="data/processed/indian_sector_data_2013_2025.csv",
        help="Path to price data CSV"
    )
    
    # Strategy parameters
    parser.add_argument("--top-n", type=int, default=4, help="Number of top sectors to hold")
    parser.add_argument("--max-weight", type=float, default=0.35, help="Max weight per sector")
    parser.add_argument("--min-weight", type=float, default=0.02, help="Min weight per sector")
    parser.add_argument("--trend-filter", type=bool, default=True, help="Use trend filter")
    parser.add_argument("--rs-lookback", type=int, default=52, help="RS-Ratio lookback period")
    parser.add_argument("--mom-lookback", type=int, default=12, help="Momentum lookback period")
    parser.add_argument("--vol-window", type=int, default=26, help="Volatility window for weighting")
    
    # Risk-free rate
    parser.add_argument("--risk-free-rate", type=float, default=0.05, help="Annual risk-free rate")
    
    # Walk-forward parameters
    parser.add_argument("--train-window", type=int, default=504, help="Training window (days)")
    parser.add_argument("--test-window", type=int, default=63, help="Test window (days)")
    parser.add_argument("--step-size", type=int, default=63, help="Step size between folds")
    
    args = parser.parse_args()
    
    if args.mode == "backtest":
        run_backtest(args)
    else:
        run_walkforward(args)


if __name__ == "__main__":
    main()
