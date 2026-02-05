"""
Base Strategy Protocol

All sector rotation strategies should implement this interface.
"""
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional

import pandas as pd


@dataclass
class StrategyConfig:
    """Base configuration for strategies. Subclass for strategy-specific params."""
    rebalance_frequency: str = "W-FRI"  # Weekly on Friday
    max_sector_weight: float = 0.35
    min_sector_weight: float = 0.02
    top_n_sectors: int = 4


class Strategy(ABC):
    """
    Abstract base class for sector rotation strategies.
    
    A strategy takes price data and outputs sector weight allocations.
    """
    
    def __init__(self, config: Optional[StrategyConfig] = None):
        self.config = config or StrategyConfig()
        self._is_fitted = False
    
    @abstractmethod
    def fit(self, prices: pd.DataFrame, benchmark: pd.Series) -> "Strategy":
        """
        Fit the strategy on historical price data.
        
        Args:
            prices: DataFrame with sector prices (columns = sectors, index = dates)
            benchmark: Series with benchmark prices
            
        Returns:
            self for method chaining
        """
        pass
    
    @abstractmethod
    def predict_weights(self, prices: pd.DataFrame, date: pd.Timestamp) -> pd.Series:
        """
        Predict sector weights for a given date.
        
        Args:
            prices: DataFrame with sector prices up to and including `date`
            date: The date for which to predict weights
            
        Returns:
            Series with sector weights (index = sector names, values = weights)
        """
        pass
    
    def generate_weight_schedule(
        self, 
        prices: pd.DataFrame, 
        benchmark: pd.Series,
        start_date: Optional[pd.Timestamp] = None,
        end_date: Optional[pd.Timestamp] = None
    ) -> pd.DataFrame:
        """
        Generate a full weight schedule over a date range.
        
        Args:
            prices: DataFrame with sector prices
            benchmark: Series with benchmark prices
            start_date: Start of the weight schedule (defaults to first valid date)
            end_date: End of the weight schedule (defaults to last date)
            
        Returns:
            DataFrame with weights for each date (index = dates, columns = sectors)
        """
        if not self._is_fitted:
            self.fit(prices, benchmark)
        
        idx = prices.index
        if start_date is not None:
            idx = idx[idx >= start_date]
        if end_date is not None:
            idx = idx[idx <= end_date]
        
        # Get rebalance dates based on frequency
        rebalance_dates = pd.date_range(
            start=idx[0], 
            end=idx[-1], 
            freq=self.config.rebalance_frequency
        )
        rebalance_dates = rebalance_dates[rebalance_dates.isin(idx)]
        
        weights = pd.DataFrame(0.0, index=idx, columns=prices.columns)
        current_weights = pd.Series(0.0, index=prices.columns)
        
        for date in idx:
            if date in rebalance_dates:
                current_weights = self.predict_weights(prices.loc[:date], date)
            weights.loc[date] = current_weights
        
        return weights
    
    @property
    def is_fitted(self) -> bool:
        return self._is_fitted
    
    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(config={self.config})"
