# Data Quality Documentation

This document describes known data anomalies and how they are handled.

> **Resolution**: To avoid this issue entirely, the data download script now starts from **2017-01-01** instead of 2013-01-01.

---

## Anomaly: IT Sector Index Jump (2016-12-12)

### Description

The Nifty IT index (`^CNXIT`) from Yahoo Finance contains a discontinuity on **2016-12-12** where the price jumps from ~1,088 to ~9,953—an apparent **814.9% return** in a single day.

This is a **Yahoo Finance data quality issue**, likely caused by:
- Missing stock split adjustment
- Corporate action not applied retroactively
- Index reconstitution artifacts

### Impact

Since the benchmark is calculated as the equal-weighted average of all sector returns, this spurious jump caused:
- **79.8% single-day benchmark return** on 2016-12-12
- Distorted performance metrics (inflated returns, incorrect alpha/beta)
- Incorrect relative strength calculations for that period

### Resolution

The `data_download.py` script includes automatic anomaly detection and correction:

```python
def detect_and_fix_anomalies(df, max_daily_return=0.5):
    """
    Detect jumps >50% and scale historical data to fix discontinuities.
    """
```

**Fix applied**: All IT sector prices before 2016-12-12 are multiplied by the jump ratio (9.15) to remove the discontinuity.

### Verification

After fix, IT prices around the anomaly date:
```
Date          IT
2016-12-09    9953.9  (was 1088.0, now scaled)
2016-12-12    9953.9  (original)
2016-12-13   10058.6  (original)
```

Benchmark daily returns now range from -11% to +8% (normal).

---

## Additional Safeguards

1. **Anomaly detection**: Any daily return >50% is flagged and corrected
2. **Return capping**: Individual sector returns are capped at ±20% before benchmark calculation
3. **Forward fill**: Missing values are forward-filled to prevent spurious zero-returns

---

## Re-running Data Download

To regenerate clean data:

```bash
conda run -n torch python data/data_download.py
```

Output should show:
```
Checking for data anomalies...
  ANOMALY: IT on 2016-12-12: 814.9% return (ratio: 9.15)
    FIXED: Scaled IT data before 2016-12-12 by 9.1488
Fixed 1 anomalies.
```
