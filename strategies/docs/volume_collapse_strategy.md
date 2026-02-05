# Volume Collapse Strategy

## Overview

The Volume Collapse strategy extends RRG with a **geometric volume–based regime filter** that detects market crises by measuring cross-sectional dimensional collapse.

---

## Core Concept

During crises, all sectors move together → rotation signals become unreliable.

**Geometric Volume** measures the "spread" of sector return vectors:
- **High volume** → sectors diverge → rotation valid
- **Low volume** → sectors collapse → reduce exposure

---

## Mathematical Definition

Given a rolling window of sector returns `R` (W × N matrix):

```
Geometric Volume = √|det(R^T · R)|
```

Where returns are normalized by their column norms before computing the Gram matrix.

---

## Parameters

### Inherited from RRG
| Parameter | Default | Description |
|-----------|---------|-------------|
| `rs_lookback` | 52 | RS-Ratio lookback |
| `momentum_lookback` | 12 | Momentum lookback |
| `top_n_sectors` | 4 | Sectors to hold |

### Volume Collapse Specific
| Parameter | Default | Description |
|-----------|---------|-------------|
| `vol_window` | 60 | Rolling window for volume |
| `vol_percentile` | 0.15 | Collapse threshold percentile |
| `risk_reduction_factor` | 0.5 | Binary mode reduction |
| `min_exposure` | 0.2 | Minimum allocation floor |
| `smooth_scaling` | True | Continuous vs binary scaling |

---

## Regime Detection

### Threshold
```
threshold = rolling_quantile(volume, vol_percentile)
```

### Scaling (Continuous Mode)
```
scale = clip(volume / threshold, min_exposure, 1.0)
```

### Scaling (Binary Mode)
```
scale = 0.5 if volume < threshold else 1.0
```

---

## Regime Labels

| Condition | Regime | Action |
|-----------|--------|--------|
| Volume < Threshold | **Collapse** | Reduce allocation |
| Threshold ≤ Volume < 2×Threshold | **Transitional** | Normal allocation |
| Volume ≥ 2×Threshold | **Diversified** | Full allocation |

---

## Usage

```python
from strategies import VolumeCollapseStrategy, VolumeCollapseConfig

config = VolumeCollapseConfig(
    top_n_sectors=5,
    vol_window=60,
    vol_percentile=0.15,
    smooth_scaling=True,
)
strategy = VolumeCollapseStrategy(config)
strategy.fit(prices, benchmark)

# Check regime
regime = strategy.get_current_regime(date)
print(f"Regime: {regime['regime']} | Scale: {regime['allocation_pct']:.1f}%")

# Get weights
weights = strategy.predict_weights(prices, date)
```

---

## Key Methods

| Method | Returns |
|--------|---------|
| `get_current_regime(date)` | dict with volume, threshold, scale, regime label |
| `get_volume_history()` | DataFrame with volume time series |
| `get_quadrant_classification(date)` | RRG quadrant info (inherited) |

---

## References

- Sudjianto (2026): "The Geometry of a Crash: Why Market Volume Beats Probabilistic Models"
