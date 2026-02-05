# Design Document

## Geometric Volume–Filtered Sector Rotation Strategy

### Version

v1.0 (Research / Backtest)

### Author

—

### Status

Approved for implementation

---

## 1. Objective

Design and implement a **sector rotation trading strategy** that extends an existing **Relative Rotation Graph (RRG)** framework with a **geometric volume–based regime filter**.

The regime filter suppresses false rotation signals during systemic market crises by detecting **cross-sectional dimensional collapse** in sector returns.

---

## 2. Background & Motivation

Traditional sector rotation strategies (including RRG-based momentum systems) assume that:

* Cross-sectional dispersion exists
* Relative performance signals are meaningful

During market crises, these assumptions fail:

* Sector returns become highly correlated
* Relative momentum signals degrade
* Rotation increases drawdowns and turnover

The geometric volume framework (Sudjianto, 2026) provides a **continuous, multivariate regime signal** that detects when the market’s return space collapses into fewer effective dimensions.

This strategy integrates that signal as a **risk and allocation controller**, not as a return predictor.

---

## 3. System Overview

### High-Level Architecture

```
Prices
  ├── Returns
  │     ├── RRG (RS-Ratio, RS-Momentum)
  │     └── Inverse Volatility Weights
  │
  └── Geometric Volume (Regime Filter)
            └── Risk Scaling / Entry Gating
                    └── Final Portfolio Weights
```

---

## 4. Design Principles

1. **Composition over inheritance**
   Regime filters are orthogonal to rotation logic.

2. **Price completeness**
   Strategy relies only on index prices and derived returns.

3. **Continuous risk control**
   Avoid binary regime switches.

4. **No look-ahead bias**
   All signals are computed with proper lagging.

5. **Extensibility**
   Architecture must support future regime filters.

---

## 5. Data Requirements

### Mandatory

* Weekly sector index prices
* Weekly benchmark index price
* Continuous historical coverage

### Derived

* Weekly returns
* Rolling volatility
* Cross-sectional dispersion

### Optional (not required for v1)

* Volume
* Breadth

---

## 6. Strategy Components

### 6.1 Base Strategy: RRG

The base strategy computes:

* Relative Strength (sector / benchmark)
* RS-Ratio (normalized relative strength)
* RS-Momentum (rate of change of RS-Ratio)
* Sector ranking by RS-Momentum
* Selection of top-N sectors
* Volatility-adjusted weights (inverse volatility)

This component is implemented and reused without modification.

---

### 6.2 Regime Filter: Geometric Volume

#### Definition

Let ( R_t \in \mathbb{R}^{W \times N} ) be a rolling window of normalized sector returns.

The geometric volume is defined as:

[
V_t = \sqrt{\left|\det(R_t^\top R_t)\right|}
]

This measures the **effective dimensionality** of cross-sectional returns.

---

#### Interpretation

| Volume Regime | Market State      | Rotation Validity |
| ------------- | ----------------- | ----------------- |
| High          | Diversified       | Strong            |
| Medium        | Transitional      | Moderate          |
| Low           | Crisis / Collapse | Weak              |

---

## 7. Configuration Design

### 7.1 Configuration Objects

#### RRGConfig (existing)

* Momentum lookback
* RS lookback
* Number of sectors
* Volatility window

#### VolumeCollapseConfig (new)

```text
VolumeCollapseConfig
├── rrg_config: RRGConfig
├── vol_window: int (default = 60 weeks)
├── vol_percentile: float (default = 0.15)
├── risk_reduction_factor: float (default = 0.5)
├── min_exposure: float (default = 0.2)
├── smooth_scaling: bool (default = True)
```

**Note:**
`VolumeCollapseConfig` **does not inherit** from `RRGConfig`.

---

## 8. Strategy Class Design

### 8.1 VolumeCollapseStrategy

**Inheritance**

```text
VolumeCollapseStrategy → RRGStrategy
```

**Responsibilities**

* Delegate all rotation logic to `RRGStrategy`
* Compute and store geometric volume time series
* Apply regime-aware allocation controls

---

### 8.2 Lifecycle Methods

#### `fit(prices)`

Responsibilities:

1. Call `super().fit(prices)`
2. Compute weekly returns
3. Compute geometric volume time series
4. Cache regime thresholds

No portfolio logic occurs here.

---

#### `predict_weights(t)`

Responsibilities:

1. Retrieve base RRG weights
2. Retrieve geometric volume at time `t`
3. Compute risk scaling factor
4. Apply scaling and exposure floor
5. Return final weights

---

## 9. Regime Logic

### 9.1 Threshold Computation

```text
collapse_threshold = rolling_percentile(volume, vol_percentile)
```

---

### 9.2 Risk Scaling (Continuous)

If `smooth_scaling = True`:

[
\text{scale}*t = \text{clip}\left(
\frac{V_t}{V*{\text{threshold}}},
\text{min_exposure},
1.0
\right)
]

Else (fallback binary mode):

```text
scale = risk_reduction_factor if V_t < threshold else 1.0
scale = max(scale, min_exposure)
```

---

### 9.3 Entry Gating (Optional Enhancement)

During collapse regimes:

* Disallow new sector entries
* Allow only existing positions to persist

This reduces churn during crises.

---

## 10. Portfolio Construction Rules

1. Top-N sectors by RS-Momentum
2. Volatility-adjusted weighting
3. Exposure scaling via geometric volume
4. Minimum exposure enforced
5. One-period lag on weights

---

## 11. Execution & CLI Design

### CLI Options

```bash
--strategy rrg
--regime-filter geometric-volume
--vol-window 60
--vol-percentile 0.15
--min-exposure 0.2
```

This supports future filters without changing base logic.

---

## 12. Verification & Testing Plan

### 12.1 Automated Verification

Script: `verify_gvs.py`

Metrics:

* CAGR
* Max Drawdown
* Sharpe Ratio
* Sortino Ratio
* Turnover
* % Time in Collapse Regime
* Average Exposure During Collapse

---

### 12.2 Manual Validation

* Confirm exposure reduction during known crises
* Inspect rotation stability
* Compare baseline RRG vs VolumeCollapse

---

## 13. Risks & Mitigations

| Risk                   | Mitigation               |
| ---------------------- | ------------------------ |
| Overfitting thresholds | Percentile-based scaling |
| Hidden cash timing     | Minimum exposure floor   |
| Noisy regime detection | Continuous scaling       |
| Architecture rigidity  | Composition-based config |

---

## 14. Future Extensions

* Volatility-based regime filters
* Macro factor overlays
* Multi-resolution volume (weekly + monthly)
* Asset-class rotation beyond equities

---

## 15. Summary

This design introduces a **geometrically grounded, continuous regime filter** into a sector rotation framework without compromising modularity or statistical integrity.

The resulting system:

* Avoids crisis-era false rotation
* Preserves alpha in normal regimes
* Remains extensible and research-friendly

---

If you want next, I can:

* Translate this doc into **class skeleton code**
* Generate **unit tests**
* Or produce a **hyperparameter safety guide**

Just tell me.
