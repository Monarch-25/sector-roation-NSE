"""
Walk-Forward Validation for Sector Rotation Strategies

Implements rolling window validation for out-of-sample performance testing.
"""
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Any

import numpy as np
import pandas as pd

from strategies.base import Strategy
from backtesting.backtester import Backtester, BacktestResult


@dataclass
class WalkForwardConfig:
    """Configuration for walk-forward validation."""
    # Training window size (in trading days)
    train_window: int = 252 * 2  # 2 years
    
    # Test window size (in trading days)
    test_window: int = 63  # ~3 months (quarterly)
    
    # Step size between windows
    step_size: int = 63  # Move forward by test_window
    
    # Minimum training window (for initial periods)
    min_train_window: int = 252  # 1 year minimum


@dataclass
class WalkForwardResult:
    """Results from walk-forward validation."""
    # Individual fold results
    fold_results: List[BacktestResult]
    fold_periods: List[Dict[str, pd.Timestamp]]
    
    # Aggregated out-of-sample performance
    oos_equity: pd.Series
    oos_returns: pd.Series
    
    # Aggregated metrics
    aggregate_metrics: Dict[str, float]
    
    # Per-fold metrics
    fold_metrics: pd.DataFrame


class WalkForwardValidator:
    """
    Walk-Forward Validation framework for strategy testing.
    
    This tests strategy performance in a realistic manner by:
    1. Training on historical data
    2. Testing on unseen future data
    3. Rolling forward and repeating
    
    Usage:
        strategy = RRGStrategy()
        validator = WalkForwardValidator(strategy, config)
        result = validator.validate(prices, benchmark)
    """
    
    def __init__(
        self,
        strategy: Strategy,
        config: Optional[WalkForwardConfig] = None,
        risk_free_rate: float = 0.05
    ):
        self.strategy = strategy
        self.config = config or WalkForwardConfig()
        self.risk_free_rate = risk_free_rate
    
    def validate(
        self,
        prices: pd.DataFrame,
        benchmark: pd.Series
    ) -> WalkForwardResult:
        """
        Run walk-forward validation.
        
        Args:
            prices: Full price history
            benchmark: Full benchmark price history
            
        Returns:
            WalkForwardResult with aggregated out-of-sample performance
        """
        idx = prices.index
        n = len(idx)
        
        cfg = self.config
        fold_results = []
        fold_periods = []
        
        # Generate fold boundaries
        start_pos = cfg.min_train_window
        
        while start_pos + cfg.test_window <= n:
            # Training period
            train_start = max(0, start_pos - cfg.train_window)
            train_end = start_pos - 1
            
            # Test period
            test_start = start_pos
            test_end = min(start_pos + cfg.test_window - 1, n - 1)
            
            train_dates = idx[train_start:train_end + 1]
            test_dates = idx[test_start:test_end + 1]
            
            # Run fold
            result = self._run_fold(
                prices, benchmark,
                train_dates, test_dates
            )
            
            fold_results.append(result)
            fold_periods.append({
                'train_start': train_dates[0],
                'train_end': train_dates[-1],
                'test_start': test_dates[0],
                'test_end': test_dates[-1]
            })
            
            # Move forward
            start_pos += cfg.step_size
        
        # Aggregate results
        return self._aggregate_results(fold_results, fold_periods, benchmark)
    
    def _run_fold(
        self,
        prices: pd.DataFrame,
        benchmark: pd.Series,
        train_dates: pd.DatetimeIndex,
        test_dates: pd.DatetimeIndex
    ) -> BacktestResult:
        """Run a single walk-forward fold."""
        # Get training data
        train_prices = prices.loc[train_dates]
        train_benchmark = benchmark.loc[train_dates]
        
        # Fit strategy on training data
        # Create a fresh strategy instance to avoid state leakage
        strategy_copy = self.strategy.__class__(self.strategy.config)
        strategy_copy.fit(train_prices, train_benchmark)
        
        # Run backtest on test period
        test_prices = prices.loc[test_dates]
        test_benchmark = benchmark.loc[test_dates]
        
        backtester = Backtester(
            strategy_copy,
            risk_free_rate=self.risk_free_rate
        )
        
        # For test, we need full data up to test end for proper weight computation
        full_prices = prices.loc[:test_dates[-1]]
        full_benchmark = benchmark.loc[:test_dates[-1]]
        
        result = backtester.run(
            full_prices, full_benchmark,
            start_date=test_dates[0],
            end_date=test_dates[-1]
        )
        
        return result
    
    def _aggregate_results(
        self,
        fold_results: List[BacktestResult],
        fold_periods: List[Dict],
        benchmark: pd.Series
    ) -> WalkForwardResult:
        """Aggregate results across all folds."""
        # Concatenate OOS returns
        all_oos_returns = []
        for result in fold_results:
            all_oos_returns.append(result.strategy_returns)
        
        oos_returns = pd.concat(all_oos_returns)
        oos_returns = oos_returns[~oos_returns.index.duplicated(keep='first')]
        oos_returns = oos_returns.sort_index()
        
        # Compute OOS equity curve
        oos_equity = 100 * (1 + oos_returns).cumprod()
        
        # Compute aggregate metrics using the Backtester's metric calculations
        backtester = Backtester(
            self.strategy,
            risk_free_rate=self.risk_free_rate
        )
        
        # Get benchmark returns for the OOS period
        bench_returns = benchmark.pct_change().loc[oos_returns.index].fillna(0)
        bench_equity = 100 * (1 + bench_returns).cumprod()
        
        aggregate_metrics = backtester._calculate_metrics(
            oos_returns, bench_returns, oos_equity, bench_equity
        )
        
        # Collect per-fold metrics
        fold_metric_records = []
        for i, (result, period) in enumerate(zip(fold_results, fold_periods)):
            record = {
                'fold': i + 1,
                'test_start': period['test_start'],
                'test_end': period['test_end'],
                **result.metrics
            }
            fold_metric_records.append(record)
        
        fold_metrics = pd.DataFrame(fold_metric_records)
        
        return WalkForwardResult(
            fold_results=fold_results,
            fold_periods=fold_periods,
            oos_equity=oos_equity,
            oos_returns=oos_returns,
            aggregate_metrics=aggregate_metrics,
            fold_metrics=fold_metrics
        )
    
    def print_report(self, result: WalkForwardResult) -> None:
        """Print walk-forward validation report."""
        m = result.aggregate_metrics
        
        print("=" * 70)
        print("WALK-FORWARD VALIDATION REPORT")
        print("=" * 70)
        
        print(f"\nNumber of Folds: {len(result.fold_results)}")
        print(f"Total OOS Period: {result.oos_returns.index[0]} to {result.oos_returns.index[-1]}")
        
        print("\n--- Aggregate Out-of-Sample Performance ---")
        print(f"Total Return:          {m['total_return']:>10.2%}")
        print(f"CAGR:                  {m['cagr']:>10.2%}")
        print(f"Sharpe Ratio:          {m['sharpe_ratio']:>10.3f}")
        print(f"Sortino Ratio:         {m['sortino_ratio']:>10.3f}")
        print(f"Max Drawdown:          {m['max_drawdown']:>10.2%}")
        print(f"Alpha (annual):        {m['alpha']:>10.2%}")
        print(f"Information Ratio:     {m['information_ratio']:>10.3f}")
        
        print("\n--- Per-Fold Summary Statistics ---")
        fold_df = result.fold_metrics
        
        key_metrics = ['total_return', 'sharpe_ratio', 'max_drawdown']
        for metric in key_metrics:
            if metric in fold_df.columns:
                values = fold_df[metric]
                print(f"\n{metric}:")
                print(f"  Mean:   {values.mean():.4f}")
                print(f"  Std:    {values.std():.4f}")
                print(f"  Min:    {values.min():.4f}")
                print(f"  Max:    {values.max():.4f}")
        
        # Consistency
        if 'total_return' in fold_df.columns:
            positive_folds = (fold_df['total_return'] > 0).sum()
            total_folds = len(fold_df)
            print(f"\nConsistency: {positive_folds}/{total_folds} folds profitable ({100*positive_folds/total_folds:.1f}%)")
        
        print("=" * 70)
