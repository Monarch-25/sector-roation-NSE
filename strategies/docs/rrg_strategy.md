# RRG (Relative Rotation Graph) Strategy

## Overview

The RRG strategy implements sector rotation based on **relative strength analysis** using RS-Ratio and RS-Momentum with inverse-volatility weighting.

---

## Core Concept

Sectors rotate through four quadrants based on their momentum relative to a benchmark:

| Quadrant | RS-Ratio | RS-Momentum | Action |
|----------|----------|-------------|--------|
| **Leading** | ≥100 | ≥100 | Hold/Overweight |
| **Weakening** | ≥100 | <100 | Reduce |
| **Lagging** | <100 | <100 | Avoid |
| **Improving** | <100 | ≥100 | Accumulate |

---

## Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `rs_lookback` | 52 | Weeks for RS-Ratio normalization |
| `momentum_lookback` | 12 | Weeks for momentum calculation |
| `volatility_window` | 26 | Weeks for inverse-vol weighting |
| `top_n_sectors` | 4 | Number of sectors to hold |
| `max_sector_weight` | 0.35 | Maximum weight per sector |
| `use_trend_filter` | True | Only allocate if benchmark > MA |
| `trend_ma_period` | 40 | Trend filter MA period |

---

## Signal Calculation

### 1. Relative Strength
```
RS = Sector Price / Benchmark Price
```

### 2. RS-Ratio (Normalized)
```
RS-Ratio = 100 + (RS - mean) / std
```
Computed over `rs_lookback` rolling window.

### 3. RS-Momentum
```
RS-Momentum = 100 + (ROC - mean) / std
```
Where ROC is the rate of change of RS-Ratio over `momentum_lookback` periods.

---

## Weight Allocation

1. Rank sectors by RS-Momentum
2. Select top N sectors
3. Apply inverse-volatility weighting
4. Enforce min/max weight constraints
5. Apply exponential smoothing to reduce turnover

---

## Usage

```python
from strategies import RRGStrategy, RRGConfig

config = RRGConfig(
    top_n_sectors=5,
    rs_lookback=100,
    momentum_lookback=30,
)
strategy = RRGStrategy(config)
strategy.fit(prices, benchmark)
weights = strategy.predict_weights(prices, date)
```

---

## References

- StockCharts RRG methodology
- Julius de Kempenaer's original research
