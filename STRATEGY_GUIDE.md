# Strategy Development Guide

This guide explains how to create new sector rotation strategies and how the backtesting framework evaluates them.

---

## Part 1: Strategy Structure

### Required Interface

All strategies must extend the `Strategy` base class from `strategies/base.py`:

```python
from strategies.base import Strategy, StrategyConfig

class MyStrategy(Strategy):
    def fit(self, prices: pd.DataFrame, benchmark: pd.Series) -> "MyStrategy":
        """Train/initialize the strategy on historical data."""
        # Your logic here
        self._is_fitted = True
        return self
    
    def predict_weights(self, prices: pd.DataFrame, date: pd.Timestamp) -> pd.Series:
        """Return sector weights for a given date."""
        # Your logic here
        return weights  # pd.Series with sector names as index
```

### File Structure

```
strategies/
├── __init__.py           # Export your strategy here
├── base.py               # DO NOT MODIFY - base Strategy class
├── rrg_strategy.py       # Example: RRG implementation
└── my_strategy.py        # Your new strategy
```

### Configuration Pattern

Create a dataclass extending `StrategyConfig`:

```python
from dataclasses import dataclass
from strategies.base import StrategyConfig

@dataclass
class MyStrategyConfig(StrategyConfig):
    # Inherited fields (with defaults):
    # rebalance_frequency: str = "W-FRI"
    # max_sector_weight: float = 0.35
    # min_sector_weight: float = 0.02
    # top_n_sectors: int = 4
    
    # Add your custom parameters:
    lookback_period: int = 20
    momentum_threshold: float = 0.05
```

### Key Constraints

| Rule | Reason |
|------|--------|
| **No look-ahead bias** | `predict_weights(prices, date)` receives data up to `date` only |
| **Return pd.Series** | Index = sector names, Values = weights (should sum to ≤1.0) |
| **Set `_is_fitted = True`** | After `fit()` completes, set this flag |
| **Weights shift by 1 day** | Backtester applies weights on T+1 (trade on signal, realize PnL next day) |

### Template

```python
"""
My Strategy - Brief description

Implements [methodology] for sector rotation.
"""
from dataclasses import dataclass
from typing import Optional
import pandas as pd
import numpy as np

from .base import Strategy, StrategyConfig


@dataclass
class MyStrategyConfig(StrategyConfig):
    my_param: int = 20


class MyStrategy(Strategy):
    def __init__(self, config: Optional[MyStrategyConfig] = None):
        super().__init__(config or MyStrategyConfig())
        self.config: MyStrategyConfig = self.config
        # Internal state
        self._signals: Optional[pd.DataFrame] = None
    
    def fit(self, prices: pd.DataFrame, benchmark: pd.Series) -> "MyStrategy":
        # Compute signals/indicators from historical data
        self._signals = self._compute_signals(prices, benchmark)
        self._is_fitted = True
        return self
    
    def _compute_signals(self, prices: pd.DataFrame, benchmark: pd.Series) -> pd.DataFrame:
        # Your signal logic here
        pass
    
    def predict_weights(self, prices: pd.DataFrame, date: pd.Timestamp) -> pd.Series:
        if not self._is_fitted:
            raise RuntimeError("Call fit() first")
        
        sectors = prices.columns
        weights = pd.Series(0.0, index=sectors)
        
        # Your allocation logic here
        # Example: equal weight top N sectors
        selected = self._select_sectors(date)
        if len(selected) > 0:
            weights[selected] = 1.0 / len(selected)
        
        return weights
    
    def _select_sectors(self, date: pd.Timestamp) -> list:
        # Your selection logic
        pass
```

---

## Part 2: Backtesting Mechanics

### Overview

The `Backtester` class simulates strategy performance by:
1. Fitting the strategy on the full dataset
2. Generating weights for each rebalance date
3. Computing portfolio returns as weighted sum of sector returns
4. Calculating comprehensive performance metrics

### Execution Flow

```
┌─────────────────────────────────────────────────────────────┐
│  1. LOAD DATA                                               │
│     prices (sectors) + benchmark → aligned DatetimeIndex    │
└──────────────────────────────────────────────────────────────┘
                              ▼
┌─────────────────────────────────────────────────────────────┐
│  2. FIT STRATEGY                                            │
│     strategy.fit(prices, benchmark) → precompute signals    │
└──────────────────────────────────────────────────────────────┘
                              ▼
┌─────────────────────────────────────────────────────────────┐
│  3. GENERATE WEIGHT SCHEDULE                                │
│     For each rebalance_date (e.g., every Friday):           │
│       weights[date] = strategy.predict_weights(prices, date)│
│     Hold weights until next rebalance                       │
└──────────────────────────────────────────────────────────────┘
                              ▼
┌─────────────────────────────────────────────────────────────┐
│  4. COMPUTE RETURNS                                         │
│     sector_returns = prices.pct_change()                    │
│     strategy_returns = (shifted_weights * sector_returns).sum()
│                                                             │
│     ⚠️  Weights are SHIFTED by 1 day to avoid look-ahead   │
└──────────────────────────────────────────────────────────────┘
                              ▼
┌─────────────────────────────────────────────────────────────┐
│  5. CALCULATE METRICS                                       │
│     Sharpe, Sortino, Max DD, CAGR, Alpha, Beta, etc.        │
└──────────────────────────────────────────────────────────────┘
```

### Metrics Computed

| Category | Metrics |
|----------|---------|
| **Returns** | Total Return, CAGR, Excess Return |
| **Risk** | Volatility, Downside Vol, Max Drawdown |
| **Risk-Adjusted** | Sharpe, Sortino, Calmar, Information Ratio |
| **Regression** | Alpha (annualized), Beta |
| **Trade Stats** | Win Rate, Profit Factor |
| **Drawdown** | Avg Drawdown, Avg Duration, Count |

### Walk-Forward Validation

For out-of-sample testing, `WalkForwardValidator`:

```
Train Window         Test Window
[===================][======]
      2 years         3 months
                    
        Step forward by test_window
                    
[===================][======]
      2 years         3 months
```

- **train_window**: Days to train strategy (default: 504 ≈ 2 years)
- **test_window**: Days to test OOS (default: 63 ≈ 3 months)
- Aggregates all OOS returns for final metrics

### Running Backtests

```bash
# Full backtest
python run_strategy.py --mode backtest

# Walk-forward validation
python run_strategy.py --mode walkforward

# With custom strategy params
python run_strategy.py --mode backtest --top-n 3 --max-weight 0.4 --trend-filter False
```

---

## Part 3: Adding Your Strategy

1. Create `strategies/my_strategy.py` following the template above
2. Add exports to `strategies/__init__.py`:
   ```python
   from .my_strategy import MyStrategy, MyStrategyConfig
   ```
3. Update `run_strategy.py` to support your strategy (or use directly in code):
   ```python
   from strategies import MyStrategy, MyStrategyConfig
   
   strategy = MyStrategy(MyStrategyConfig(my_param=30))
   backtester = Backtester(strategy)
   result = backtester.run(prices, benchmark)
   ```

---

## Appendix: Data Format

**Input CSV** (`data/processed/indian_sector_data_2013_2025.csv`):

| Date | Auto | Finance | IT | ... | Benchmark |
|------|------|---------|----|----|-----------|
| 2017-01-02 | 5765.3 | 3142.5 | 9312.8 | ... | 1.0 |
| 2017-01-03 | 5802.1 | 3155.2 | 9285.4 | ... | 1.008 |

- **Index**: Date (datetime)
- **Columns**: Sector price indices + Benchmark
- **Benchmark**: Equal-weighted avg return of all sectors, cumulated from 1.0
