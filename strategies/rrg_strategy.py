"""
RRG (Relative Rotation Graph) Strategy

Implements sector rotation based on relative strength analysis using
RS-Ratio and RS-Momentum with inverse-volatility weighting.
"""
from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd

from .base import Strategy, StrategyConfig


@dataclass
class RRGConfig(StrategyConfig):
    """Configuration for RRG Strategy."""
    # RS-Ratio parameters
    rs_lookback: int = 52  # weeks for RS normalization
    
    # RS-Momentum parameters  
    momentum_lookback: int = 12  # weeks for momentum calculation
    
    # Volatility window for inverse-vol weighting
    volatility_window: int = 26
    
    # Trend filter: only allocate if benchmark above N-period MA
    use_trend_filter: bool = True
    trend_ma_period: int = 40
    
    # Turnover smoothing
    weight_smoothing_alpha: float = 0.3
    
    # Full allocation: always invest 100% (no cash), leftover goes to top sector
    full_allocation: bool = True


class RRGStrategy(Strategy):
    """
    Relative Rotation Graph based sector rotation strategy.
    
    Logic:
    1. Calculate relative strength of each sector vs benchmark
    2. Normalize to RS-Ratio (centered around 100)
    3. Calculate RS-Momentum as rate of change of RS-Ratio
    4. Rank sectors by RS-Momentum, select top N
    5. Weight using inverse volatility
    6. Apply trend filter and smoothing
    """
    
    def __init__(self, config: Optional[RRGConfig] = None):
        super().__init__(config or RRGConfig())
        self.config: RRGConfig = self.config
        
        # Internal state
        self._rs_ratio: Optional[pd.DataFrame] = None
        self._rs_momentum: Optional[pd.DataFrame] = None
        self._returns: Optional[pd.DataFrame] = None
        self._benchmark: Optional[pd.Series] = None
        self._prev_weights: Optional[pd.Series] = None
    
    def fit(self, prices: pd.DataFrame, benchmark: pd.Series) -> "RRGStrategy":
        """
        Fit the RRG strategy by computing RS-Ratio and RS-Momentum.
        """
        self._benchmark = benchmark
        self._returns = prices.pct_change()
        
        # Step 1: Compute relative strength
        rs = self._compute_relative_strength(prices, benchmark)
        
        # Step 2: Compute RS-Ratio (normalized)
        self._rs_ratio = self._compute_rs_ratio(rs)
        
        # Step 3: Compute RS-Momentum
        self._rs_momentum = self._compute_rs_momentum(self._rs_ratio)
        
        self._is_fitted = True
        return self
    
    def _compute_relative_strength(
        self, prices: pd.DataFrame, benchmark: pd.Series
    ) -> pd.DataFrame:
        """Calculate relative strength of each sector vs benchmark."""
        return prices.div(benchmark, axis=0)
    
    def _compute_rs_ratio(self, rs: pd.DataFrame) -> pd.DataFrame:
        """
        Normalize relative strength to RS-Ratio centered at 100.
        RS-Ratio = 100 + (RS - mean) / std
        """
        lookback = self.config.rs_lookback
        mean = rs.rolling(lookback, min_periods=lookback // 2).mean()
        std = rs.rolling(lookback, min_periods=lookback // 2).std()
        
        # Avoid division by zero
        std = std.replace(0, np.nan)
        
        rs_ratio = 100 + (rs - mean) / std
        return rs_ratio
    
    def _compute_rs_momentum(self, rs_ratio: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate RS-Momentum as normalized rate of change.
        """
        lookback = self.config.momentum_lookback
        
        # Rate of change
        roc = rs_ratio.diff(lookback)
        
        # Normalize
        mean = roc.rolling(lookback, min_periods=lookback // 2).mean()
        std = roc.rolling(lookback, min_periods=lookback // 2).std()
        std = std.replace(0, np.nan)
        
        rs_momentum = 100 + (roc - mean) / std
        return rs_momentum
    
    def _compute_inverse_vol_weights(self, date: pd.Timestamp) -> pd.Series:
        """Calculate inverse volatility weights."""
        window = self.config.volatility_window
        
        # Get returns up to date
        returns_to_date = self._returns.loc[:date].tail(window)
        
        vol = returns_to_date.std()
        vol = vol.replace(0, np.nan)
        
        inv_vol = 1 / vol
        weights = inv_vol / inv_vol.sum()
        
        return weights.fillna(0)
    
    def _apply_trend_filter(self, date: pd.Timestamp) -> bool:
        """Check if benchmark is above its moving average."""
        if not self.config.use_trend_filter:
            return True
        
        ma_period = self.config.trend_ma_period
        benchmark_to_date = self._benchmark.loc[:date]
        
        if len(benchmark_to_date) < ma_period:
            return True  # Not enough data, allow allocation
        
        current = benchmark_to_date.iloc[-1]
        ma = benchmark_to_date.rolling(ma_period).mean().iloc[-1]
        
        return current > ma
    
    def _rank_and_select(self, date: pd.Timestamp) -> pd.Index:
        """Rank sectors by RS-Momentum and select top N."""
        if date not in self._rs_momentum.index:
            # Find closest available date
            available = self._rs_momentum.index[self._rs_momentum.index <= date]
            if len(available) == 0:
                return pd.Index([])
            date = available[-1]
        
        mom = self._rs_momentum.loc[date]
        
        # Filter for improving or leading quadrants (momentum > 100)
        # or just rank by momentum
        ranked = mom.sort_values(ascending=False)
        top_n = ranked.head(self.config.top_n_sectors)
        
        return top_n.index
    
    def _apply_weight_constraints(self, weights: pd.Series) -> pd.Series:
        """Apply min/max weight constraints."""
        w = weights.copy()
        
        # Zero out very small weights
        w[w < self.config.min_sector_weight] = 0
        
        # Apply max cap iteratively
        for _ in range(10):
            over = w > self.config.max_sector_weight
            if not over.any():
                break
            
            excess = (w[over] - self.config.max_sector_weight).sum()
            w[over] = self.config.max_sector_weight
            
            under = (w > 0) & (~over)
            if under.sum() > 0:
                w[under] += excess * (w[under] / w[under].sum())
        
        # Renormalize
        if w.sum() > 0:
            w = w / w.sum()
        
        return w
    
    def _smooth_weights(self, new_weights: pd.Series) -> pd.Series:
        """Apply exponential smoothing to reduce turnover."""
        if self._prev_weights is None:
            self._prev_weights = new_weights
            return new_weights
        
        alpha = self.config.weight_smoothing_alpha
        smoothed = alpha * new_weights + (1 - alpha) * self._prev_weights
        
        # Renormalize
        if smoothed.sum() > 0:
            smoothed = smoothed / smoothed.sum()
        
        self._prev_weights = smoothed
        return smoothed
    
    def predict_weights(self, prices: pd.DataFrame, date: pd.Timestamp) -> pd.Series:
        """
        Predict sector weights for a given date.
        """
        if not self._is_fitted:
            raise RuntimeError("Strategy must be fitted before predicting weights")
        
        sectors = prices.columns
        weights = pd.Series(0.0, index=sectors)
        
        # Check trend filter (skip if full_allocation is True)
        trend_ok = self._apply_trend_filter(date)
        if not trend_ok and not self.config.full_allocation:
            return weights  # Stay in cash
        
        # Rank and select top sectors
        selected = self._rank_and_select(date)
        
        if len(selected) == 0:
            return weights
        
        # Get inverse-vol weights for selected sectors only
        inv_vol_weights = self._compute_inverse_vol_weights(date)
        
        for sector in selected:
            if sector in inv_vol_weights.index:
                weights[sector] = inv_vol_weights[sector]
        
        # Renormalize to selected sectors
        if weights.sum() > 0:
            weights = weights / weights.sum()
        
        # Apply constraints
        weights = self._apply_weight_constraints(weights)
        
        # Full allocation: give leftover to top-ranked sector
        if self.config.full_allocation and weights.sum() > 0:
            leftover = 1.0 - weights.sum()
            if leftover > 0.001:  # Meaningful leftover
                # Top sector is first in selected (highest RS-Momentum)
                top_sector = selected[0]
                weights[top_sector] += leftover
        
        # Smooth weights
        weights = self._smooth_weights(weights)
        
        return weights
    
    def get_quadrant_classification(self, date: pd.Timestamp) -> pd.DataFrame:
        """
        Classify sectors into RRG quadrants for a given date.
        
        Quadrants:
        - Leading: RS-Ratio >= 100, RS-Momentum >= 100
        - Weakening: RS-Ratio >= 100, RS-Momentum < 100
        - Improving: RS-Ratio < 100, RS-Momentum >= 100
        - Lagging: RS-Ratio < 100, RS-Momentum < 100
        """
        if not self._is_fitted:
            raise RuntimeError("Strategy must be fitted first")
        
        if date not in self._rs_ratio.index:
            available = self._rs_ratio.index[self._rs_ratio.index <= date]
            if len(available) == 0:
                return pd.DataFrame()
            date = available[-1]
        
        r = self._rs_ratio.loc[date]
        m = self._rs_momentum.loc[date]
        
        quadrant = pd.Series(index=r.index, dtype=str)
        quadrant[(r >= 100) & (m >= 100)] = "Leading"
        quadrant[(r >= 100) & (m < 100)] = "Weakening"
        quadrant[(r < 100) & (m >= 100)] = "Improving"
        quadrant[(r < 100) & (m < 100)] = "Lagging"
        
        return pd.DataFrame({
            "RS_Ratio": r,
            "RS_Momentum": m,
            "Quadrant": quadrant
        })
