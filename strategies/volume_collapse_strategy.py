"""
Geometric Volume Collapse Strategy

Extends RRG sector rotation with a geometric volume–based regime filter.
When cross-sectional returns collapse (crisis), the filter reduces allocation.
"""
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd

from .rrg_strategy import RRGStrategy, RRGConfig


@dataclass
class VolumeCollapseConfig(RRGConfig):
    """Configuration for Volume Collapse Strategy.
    
    Inherits all RRG parameters and adds geometric volume filter settings.
    """
    # Geometric volume window (weeks)
    vol_window: int = 60
    
    # Percentile threshold for collapse detection (lower = more sensitive)
    vol_percentile: float = 0.15
    
    # Risk reduction factor when in collapse regime (binary mode)
    risk_reduction_factor: float = 0.5
    
    # Minimum exposure floor (never go below this)
    min_exposure: float = 0.2
    
    # Use smooth scaling instead of binary threshold
    smooth_scaling: bool = True


class VolumeCollapseStrategy(RRGStrategy):
    """
    Geometric Volume Collapse Strategy.
    
    Extends RRG with a regime filter based on the geometric volume of
    cross-sectional sector returns. When volume collapses (crisis regime),
    allocation is reduced to avoid false rotation signals.
    
    Logic:
    1. Compute geometric volume as sqrt(|det(R^T R)|) where R is a 
       matrix of normalized rolling returns.
    2. Determine regime via rolling percentile threshold.
    3. Scale RRG-derived weights based on current regime.
    """
    
    def __init__(self, config: Optional[VolumeCollapseConfig] = None):
        super().__init__(config or VolumeCollapseConfig())
        self.config: VolumeCollapseConfig = self.config
        
        # Additional state for geometric volume
        self._geometric_volume: Optional[pd.Series] = None
        self._volume_threshold: Optional[pd.Series] = None
    
    def fit(self, prices: pd.DataFrame, benchmark: pd.Series) -> "VolumeCollapseStrategy":
        """
        Fit the strategy by computing RRG signals and geometric volume.
        """
        # Step 1: Fit base RRG strategy
        super().fit(prices, benchmark)
        
        # Step 2: Compute geometric volume time series
        returns = prices.pct_change().dropna()
        self._geometric_volume = self._compute_geometric_volume(
            returns, 
            window=self.config.vol_window
        )
        
        # Step 3: Compute rolling threshold
        self._volume_threshold = self._geometric_volume.rolling(
            window=self.config.vol_window,
            min_periods=self.config.vol_window // 2
        ).quantile(self.config.vol_percentile)
        
        return self
    
    def _compute_geometric_volume(
        self, 
        returns: pd.DataFrame, 
        window: int
    ) -> pd.Series:
        """
        Compute geometric volume of sector returns over a rolling window.
        
        Geometric volume = sqrt(|det(R^T R)|)
        where R is the matrix of normalized returns.
        
        High volume → diversified regime → rotation valid
        Low volume → crisis / collapse → rotation invalid
        """
        vols = []
        dates = returns.index[window:]
        vals = returns.values
        
        for i in range(window, len(vals)):
            wnd = vals[i - window:i]
            
            # Normalize each column (sector) by its norm
            norms = np.linalg.norm(wnd, axis=0)
            norms[norms == 0] = 1e-8  # Avoid division by zero
            normalized = wnd / norms
            
            # Gram matrix and determinant
            gram = normalized.T @ normalized
            vol = np.sqrt(np.abs(np.linalg.det(gram)))
            vols.append(vol)
        
        return pd.Series(vols, index=dates, name="geometric_volume")
    
    def _get_regime_scale(self, date: pd.Timestamp) -> float:
        """
        Compute the risk scaling factor based on geometric volume regime.
        
        Returns a value between min_exposure and 1.0.
        """
        if self._geometric_volume is None or self._volume_threshold is None:
            return 1.0
        
        # Find closest available date
        available = self._geometric_volume.index[self._geometric_volume.index <= date]
        if len(available) == 0:
            return 1.0
        
        lookup_date = available[-1]
        current_vol = self._geometric_volume.loc[lookup_date]
        threshold = self._volume_threshold.loc[lookup_date]
        
        if pd.isna(current_vol) or pd.isna(threshold) or threshold == 0:
            return 1.0
        
        if self.config.smooth_scaling:
            # Continuous scaling: scale = clip(vol / threshold, min_exposure, 1.0)
            scale = np.clip(
                current_vol / threshold,
                self.config.min_exposure,
                1.0
            )
        else:
            # Binary mode
            if current_vol < threshold:
                scale = max(self.config.risk_reduction_factor, self.config.min_exposure)
            else:
                scale = 1.0
        
        return float(scale)
    
    def predict_weights(self, prices: pd.DataFrame, date: pd.Timestamp) -> pd.Series:
        """
        Predict sector weights for a given date with regime filtering.
        """
        # Step 1: Get base RRG weights
        base_weights = super().predict_weights(prices, date)
        
        # Step 2: Get regime scale
        scale = self._get_regime_scale(date)
        
        # Step 3: Apply scaling
        scaled_weights = base_weights * scale
        
        # Step 4: Optionally redistribute leftover (if full_allocation is True)
        if self.config.full_allocation and scaled_weights.sum() > 0:
            # Keep relative proportions, but cap at scale
            # The leftover goes to "cash" (not invested)
            pass  # We don't redistribute; reduced exposure is intentional
        
        return scaled_weights
    
    def get_current_regime(self, date: pd.Timestamp) -> dict:
        """
        Get detailed regime information for a given date.
        
        Returns dict with volume, threshold, scale, and regime label.
        """
        if self._geometric_volume is None:
            return {"error": "Strategy not fitted"}
        
        available = self._geometric_volume.index[self._geometric_volume.index <= date]
        if len(available) == 0:
            return {"error": "No data available for date"}
        
        lookup_date = available[-1]
        vol = self._geometric_volume.loc[lookup_date]
        threshold = self._volume_threshold.loc[lookup_date]
        scale = self._get_regime_scale(date)
        
        # Determine regime label
        if pd.isna(vol) or pd.isna(threshold):
            regime = "Unknown"
        elif vol < threshold:
            regime = "Collapse"
        elif vol < threshold * 2:
            regime = "Transitional"
        else:
            regime = "Diversified"
        
        return {
            "date": lookup_date,
            "geometric_volume": vol,
            "threshold": threshold,
            "scale": scale,
            "regime": regime,
            "allocation_pct": scale * 100
        }
    
    def get_volume_history(self) -> pd.DataFrame:
        """
        Get the full geometric volume history for analysis/plotting.
        """
        if self._geometric_volume is None:
            raise RuntimeError("Strategy must be fitted first")
        
        return pd.DataFrame({
            "geometric_volume": self._geometric_volume,
            "threshold": self._volume_threshold
        })
