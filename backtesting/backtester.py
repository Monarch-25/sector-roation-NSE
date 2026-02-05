"""
Generic Backtester for Sector Rotation Strategies

Supports any strategy that implements the Strategy protocol.
Computes comprehensive performance metrics.
"""
from dataclasses import dataclass, field
from typing import Dict, Optional, Any

import numpy as np
import pandas as pd

from strategies.base import Strategy


@dataclass
class BacktestResult:
    """Container for backtest results."""
    # Equity curves
    strategy_equity: pd.Series
    benchmark_equity: pd.Series
    
    # Returns
    strategy_returns: pd.Series
    benchmark_returns: pd.Series
    
    # Weights over time
    weights: pd.DataFrame
    
    # Metrics
    metrics: Dict[str, float] = field(default_factory=dict)
    
    # Raw data reference
    prices: Optional[pd.DataFrame] = None
    benchmark: Optional[pd.Series] = None


class Backtester:
    """
    Generic backtester for sector rotation strategies.
    
    Usage:
        strategy = RRGStrategy(config)
        backtester = Backtester(strategy)
        result = backtester.run(prices, benchmark)
        print(result.metrics)
    """
    
    def __init__(
        self,
        strategy: Strategy,
        initial_capital: float = 100.0,
        risk_free_rate: float = 0.05,  # Annual risk-free rate
        trading_days_per_year: int = 252
    ):
        self.strategy = strategy
        self.initial_capital = initial_capital
        self.risk_free_rate = risk_free_rate
        self.trading_days_per_year = trading_days_per_year
    
    def run(
        self,
        prices: pd.DataFrame,
        benchmark: pd.Series,
        start_date: Optional[pd.Timestamp] = None,
        end_date: Optional[pd.Timestamp] = None
    ) -> BacktestResult:
        """
        Run backtest on the given data.
        
        Args:
            prices: DataFrame with sector prices
            benchmark: Series with benchmark prices
            start_date: Optional start date for backtest
            end_date: Optional end date for backtest
            
        Returns:
            BacktestResult with equity curves, weights, and metrics
        """
        # Fit strategy
        self.strategy.fit(prices, benchmark)
        
        # Generate weight schedule
        weights = self.strategy.generate_weight_schedule(
            prices, benchmark, start_date, end_date
        )
        
        # Calculate returns
        sector_returns = prices.pct_change().fillna(0)
        benchmark_returns = benchmark.pct_change().fillna(0)
        
        # Align data
        common_idx = weights.index.intersection(sector_returns.index)
        weights = weights.loc[common_idx]
        sector_returns = sector_returns.loc[common_idx]
        benchmark_returns = benchmark_returns.loc[common_idx]
        
        # Strategy returns (weighted sum)
        # Shift weights by 1 to avoid look-ahead bias
        shifted_weights = weights.shift(1).fillna(0)
        strategy_returns = (shifted_weights * sector_returns).sum(axis=1)
        
        # Equity curves
        strategy_equity = self.initial_capital * (1 + strategy_returns).cumprod()
        benchmark_equity = self.initial_capital * (1 + benchmark_returns).cumprod()
        
        # Calculate metrics
        metrics = self._calculate_metrics(
            strategy_returns, 
            benchmark_returns,
            strategy_equity,
            benchmark_equity
        )
        
        return BacktestResult(
            strategy_equity=strategy_equity,
            benchmark_equity=benchmark_equity,
            strategy_returns=strategy_returns,
            benchmark_returns=benchmark_returns,
            weights=weights,
            metrics=metrics,
            prices=prices,
            benchmark=benchmark
        )
    
    def _calculate_metrics(
        self,
        strategy_returns: pd.Series,
        benchmark_returns: pd.Series,
        strategy_equity: pd.Series,
        benchmark_equity: pd.Series
    ) -> Dict[str, float]:
        """Calculate comprehensive performance metrics."""
        metrics = {}
        
        n_days = len(strategy_returns)
        ann_factor = self.trading_days_per_year
        rf_daily = self.risk_free_rate / ann_factor
        
        # --- Returns Metrics ---
        metrics["total_return"] = self._total_return(strategy_equity)
        metrics["benchmark_total_return"] = self._total_return(benchmark_equity)
        metrics["excess_return"] = metrics["total_return"] - metrics["benchmark_total_return"]
        
        # CAGR
        metrics["cagr"] = self._cagr(strategy_equity, n_days, ann_factor)
        metrics["benchmark_cagr"] = self._cagr(benchmark_equity, n_days, ann_factor)
        
        # --- Risk Metrics ---
        metrics["volatility"] = self._annualized_volatility(strategy_returns, ann_factor)
        metrics["benchmark_volatility"] = self._annualized_volatility(benchmark_returns, ann_factor)
        
        metrics["downside_volatility"] = self._downside_volatility(strategy_returns, ann_factor)
        
        metrics["max_drawdown"] = self._max_drawdown(strategy_equity)
        metrics["benchmark_max_drawdown"] = self._max_drawdown(benchmark_equity)
        
        # --- Risk-Adjusted Metrics ---
        metrics["sharpe_ratio"] = self._sharpe_ratio(strategy_returns, rf_daily, ann_factor)
        metrics["benchmark_sharpe"] = self._sharpe_ratio(benchmark_returns, rf_daily, ann_factor)
        
        metrics["sortino_ratio"] = self._sortino_ratio(strategy_returns, rf_daily, ann_factor)
        
        metrics["calmar_ratio"] = self._calmar_ratio(metrics["cagr"], metrics["max_drawdown"])
        
        # --- Alpha/Beta ---
        alpha, beta = self._alpha_beta(strategy_returns, benchmark_returns, rf_daily, ann_factor)
        metrics["alpha"] = alpha
        metrics["beta"] = beta
        
        # Information ratio
        metrics["information_ratio"] = self._information_ratio(
            strategy_returns, benchmark_returns, ann_factor
        )
        
        # --- Trade Statistics ---
        metrics["win_rate"] = self._win_rate(strategy_returns)
        metrics["profit_factor"] = self._profit_factor(strategy_returns)
        
        # --- Drawdown Statistics ---
        dd_stats = self._drawdown_statistics(strategy_equity)
        metrics.update(dd_stats)
        
        return metrics
    
    def _total_return(self, equity: pd.Series) -> float:
        """Calculate total return."""
        return (equity.iloc[-1] / equity.iloc[0]) - 1
    
    def _cagr(self, equity: pd.Series, n_days: int, ann_factor: int) -> float:
        """Calculate Compound Annual Growth Rate."""
        total_return = equity.iloc[-1] / equity.iloc[0]
        years = n_days / ann_factor
        if years <= 0:
            return 0.0
        return (total_return ** (1 / years)) - 1
    
    def _annualized_volatility(self, returns: pd.Series, ann_factor: int) -> float:
        """Calculate annualized volatility."""
        return returns.std() * np.sqrt(ann_factor)
    
    def _downside_volatility(self, returns: pd.Series, ann_factor: int) -> float:
        """Calculate downside deviation (volatility of negative returns)."""
        negative_returns = returns[returns < 0]
        if len(negative_returns) == 0:
            return 0.0
        return negative_returns.std() * np.sqrt(ann_factor)
    
    def _max_drawdown(self, equity: pd.Series) -> float:
        """Calculate maximum drawdown."""
        rolling_max = equity.cummax()
        drawdown = (equity / rolling_max) - 1
        return drawdown.min()
    
    def _sharpe_ratio(self, returns: pd.Series, rf_daily: float, ann_factor: int) -> float:
        """Calculate Sharpe Ratio."""
        excess_returns = returns - rf_daily
        if excess_returns.std() == 0:
            return 0.0
        return (excess_returns.mean() / excess_returns.std()) * np.sqrt(ann_factor)
    
    def _sortino_ratio(self, returns: pd.Series, rf_daily: float, ann_factor: int) -> float:
        """Calculate Sortino Ratio."""
        excess_returns = returns - rf_daily
        downside = returns[returns < 0]
        
        if len(downside) == 0 or downside.std() == 0:
            return float('inf') if excess_returns.mean() > 0 else 0.0
        
        downside_std = downside.std() * np.sqrt(ann_factor)
        return (excess_returns.mean() * ann_factor) / downside_std
    
    def _calmar_ratio(self, cagr: float, max_dd: float) -> float:
        """Calculate Calmar Ratio (CAGR / Max Drawdown)."""
        if max_dd == 0:
            return float('inf') if cagr > 0 else 0.0
        return cagr / abs(max_dd)
    
    def _alpha_beta(
        self, 
        strategy_returns: pd.Series, 
        benchmark_returns: pd.Series,
        rf_daily: float,
        ann_factor: int
    ) -> tuple:
        """Calculate Alpha and Beta."""
        excess_strat = strategy_returns - rf_daily
        excess_bench = benchmark_returns - rf_daily
        
        # Covariance / Variance
        cov = excess_strat.cov(excess_bench)
        var = excess_bench.var()
        
        if var == 0:
            return 0.0, 0.0
        
        beta = cov / var
        alpha_daily = excess_strat.mean() - beta * excess_bench.mean()
        alpha_annual = alpha_daily * ann_factor
        
        return alpha_annual, beta
    
    def _information_ratio(
        self, 
        strategy_returns: pd.Series, 
        benchmark_returns: pd.Series,
        ann_factor: int
    ) -> float:
        """Calculate Information Ratio."""
        excess = strategy_returns - benchmark_returns
        tracking_error = excess.std() * np.sqrt(ann_factor)
        
        if tracking_error == 0:
            return 0.0
        
        return (excess.mean() * ann_factor) / tracking_error
    
    def _win_rate(self, returns: pd.Series) -> float:
        """Calculate win rate (% of positive return days)."""
        if len(returns) == 0:
            return 0.0
        return (returns > 0).sum() / len(returns)
    
    def _profit_factor(self, returns: pd.Series) -> float:
        """Calculate profit factor (gross profit / gross loss)."""
        gains = returns[returns > 0].sum()
        losses = abs(returns[returns < 0].sum())
        
        if losses == 0:
            return float('inf') if gains > 0 else 0.0
        
        return gains / losses
    
    def _drawdown_statistics(self, equity: pd.Series) -> Dict[str, Any]:
        """Calculate detailed drawdown statistics."""
        rolling_max = equity.cummax()
        drawdown = (equity / rolling_max) - 1
        
        # Find drawdown periods
        is_dd = drawdown < 0
        dd_starts = is_dd & ~is_dd.shift(1).fillna(False)
        dd_ends = ~is_dd & is_dd.shift(1).fillna(False)
        
        dd_periods = []
        start_idx = None
        
        for i, (date, is_start) in enumerate(dd_starts.items()):
            if is_start:
                start_idx = date
            if dd_ends.iloc[i] and start_idx is not None:
                dd_periods.append({
                    'start': start_idx,
                    'end': date,
                    'depth': drawdown.loc[start_idx:date].min()
                })
                start_idx = None
        
        if len(dd_periods) == 0:
            return {
                "avg_drawdown": 0.0,
                "avg_drawdown_duration_days": 0,
                "num_drawdowns": 0
            }
        
        depths = [p['depth'] for p in dd_periods]
        durations = [(p['end'] - p['start']).days for p in dd_periods]
        
        return {
            "avg_drawdown": np.mean(depths),
            "avg_drawdown_duration_days": np.mean(durations),
            "num_drawdowns": len(dd_periods)
        }
    
    def print_report(self, result: BacktestResult) -> None:
        """Print a formatted performance report."""
        m = result.metrics
        
        print("=" * 60)
        print("BACKTEST PERFORMANCE REPORT")
        print("=" * 60)
        
        print("\n--- Returns ---")
        print(f"Total Return:          {m['total_return']:>10.2%}")
        print(f"Benchmark Return:      {m['benchmark_total_return']:>10.2%}")
        print(f"Excess Return:         {m['excess_return']:>10.2%}")
        print(f"CAGR:                  {m['cagr']:>10.2%}")
        print(f"Benchmark CAGR:        {m['benchmark_cagr']:>10.2%}")
        
        print("\n--- Risk ---")
        print(f"Volatility:            {m['volatility']:>10.2%}")
        print(f"Downside Vol:          {m['downside_volatility']:>10.2%}")
        print(f"Max Drawdown:          {m['max_drawdown']:>10.2%}")
        print(f"Benchmark Max DD:      {m['benchmark_max_drawdown']:>10.2%}")
        
        print("\n--- Risk-Adjusted ---")
        print(f"Sharpe Ratio:          {m['sharpe_ratio']:>10.3f}")
        print(f"Benchmark Sharpe:      {m['benchmark_sharpe']:>10.3f}")
        print(f"Sortino Ratio:         {m['sortino_ratio']:>10.3f}")
        print(f"Calmar Ratio:          {m['calmar_ratio']:>10.3f}")
        print(f"Information Ratio:     {m['information_ratio']:>10.3f}")
        
        print("\n--- Alpha/Beta ---")
        print(f"Alpha (annual):        {m['alpha']:>10.2%}")
        print(f"Beta:                  {m['beta']:>10.3f}")
        
        print("\n--- Trade Stats ---")
        print(f"Win Rate:              {m['win_rate']:>10.2%}")
        print(f"Profit Factor:         {m['profit_factor']:>10.3f}")
        
        print("\n--- Drawdown Stats ---")
        print(f"Avg Drawdown:          {m['avg_drawdown']:>10.2%}")
        print(f"Avg DD Duration:       {m['avg_drawdown_duration_days']:>10.0f} days")
        print(f"Num Drawdowns:         {m['num_drawdowns']:>10.0f}")
        
        print("=" * 60)
