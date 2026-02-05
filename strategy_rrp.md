1. Core Design Philosophy (Important Correction)

Your intuition is half-right, half-dangerous.

RRG must be built on relative strength, not raw pct-change

Using returns instead of prices inside RRG breaks the geometry

BUT returns are excellent for portfolio weighting, volatility control, and risk metrics

Correct architecture:

Prices → Relative Strength → RRG (RS-Ratio, RS-Momentum)
Returns → Volatility, Sharpe, Sortino, Risk Scaling


Do NOT replace price with returns in RS computation.

2. Data Assumptions

You already have:

prices: pd.DataFrame
index: datetime (weekly)
columns: ['NIFTY', 'BANK', 'IT', 'PHARMA', 'AUTO', ...]
values: index levels


Weekly frequency is mandatory.

3. Step 1 — Relative Strength (RRG Base)
import pandas as pd
import numpy as np

def compute_relative_strength(prices, benchmark):
    rs = prices.div(prices[benchmark], axis=0)
    return rs

4. Step 2 — JdK RS-Ratio (X-axis)

Standard RRG uses normalized relative strength.

def compute_rs_ratio(rs, lookback=52):
    mean = rs.rolling(lookback).mean()
    std = rs.rolling(lookback).std()
    rs_ratio = 100 + (rs - mean) / std
    return rs_ratio

5. Step 3 — JdK RS-Momentum (Y-axis)

Momentum of relative strength.

def compute_rs_momentum(rs_ratio, momentum_lookback=12):
    roc = rs_ratio.diff(momentum_lookback)
    mean = roc.rolling(momentum_lookback).mean()
    std = roc.rolling(momentum_lookback).std()
    rs_momentum = 100 + (roc - mean) / std
    return rs_momentum

6. Step 4 — Quadrant Classification
def classify_quadrant(rs_ratio, rs_momentum):
    conditions = {
        "Leading": (rs_ratio >= 100) & (rs_momentum >= 100),
        "Weakening": (rs_ratio >= 100) & (rs_momentum < 100),
        "Improving": (rs_ratio < 100) & (rs_momentum >= 100),
        "Lagging": (rs_ratio < 100) & (rs_momentum < 100),
    }

    quadrant = pd.DataFrame(index=rs_ratio.index, columns=rs_ratio.columns)
    for q, cond in conditions.items():
        quadrant[cond] = q

    return quadrant

7. Step 5 — Momentum Ranking (Top-4 Selection)

Instead of quadrant only, rank by RS-Momentum.

def rank_sectors(rs_momentum, top_n=4):
    return rs_momentum.rank(axis=1, ascending=False) <= top_n


This gives boolean inclusion mask.

8. Step 6 — Return Space (Your Normal Distribution Idea)

Now returns are used correctly.

def compute_returns(prices):
    return prices.pct_change()

9. Step 7 — Volatility-Adjusted Weights (Critical Robustness)

Instead of equal weights → inverse volatility.

def inverse_vol_weights(returns, window=26):
    vol = returns.rolling(window).std()
    inv_vol = 1 / vol
    weights = inv_vol.div(inv_vol.sum(axis=1), axis=0)
    return weights

10. Step 8 — Final Portfolio Weights
def construct_portfolio_weights(
    prices,
    benchmark="NIFTY",
    top_n=4
):
    rs = compute_relative_strength(prices, benchmark)
    rs_ratio = compute_rs_ratio(rs)
    rs_momentum = compute_rs_momentum(rs_ratio)

    inclusion = rank_sectors(rs_momentum, top_n=top_n)

    returns = compute_returns(prices)
    vol_weights = inverse_vol_weights(returns)

    weights = inclusion * vol_weights
    weights = weights.div(weights.sum(axis=1), axis=0)

    return weights.shift(1)  # avoid lookahead

11. Step 9 — Portfolio Returns
def portfolio_returns(weights, returns):
    return (weights * returns).sum(axis=1)

12. Performance Metrics (Sharpe & Sortino)
Sharpe Ratio
def sharpe_ratio(returns, rf=0.0, freq=52):
    excess = returns - rf / freq
    return np.sqrt(freq) * excess.mean() / excess.std()

Sortino Ratio
def sortino_ratio(returns, rf=0.0, freq=52):
    downside = returns[returns < 0]
    downside_std = downside.std()
    excess = returns.mean() - rf / freq
    return np.sqrt(freq) * excess / downside_std

13. Additional Robustness Features (Strongly Recommended)

You should instruct the LLM to include:

1. Trend Filter

Only allocate if benchmark above 200-DMA.

trend = prices['NIFTY'] > prices['NIFTY'].rolling(40).mean()
weights = weights.mul(trend, axis=0)

2. Max Sector Cap

Prevent over-concentration.

weights = weights.clip(upper=0.35)
weights = weights.div(weights.sum(axis=1), axis=0)

3. Turnover Penalty

Discourage frequent flips.

weights = weights.ewm(alpha=0.3).mean()

4. Drawdown Control

Exit if portfolio drawdown > X%.

14. Final Strategy Summary (LLM Prompt-Ready)

You can feed this to an LLM as:

Build a weekly sector rotation strategy using Relative Rotation Graphs (RRG).
Use sector index prices relative to NIFTY to compute RS-Ratio and RS-Momentum.
Rank sectors by RS-Momentum and select top 4.
Allocate weights using inverse volatility on weekly returns.
Apply trend filter on benchmark, cap sector exposure, smooth weights, and avoid lookahead bias.
Evaluate performance using CAGR, Max Drawdown, Sharpe, and Sortino.

15. Why This Strategy Is Statistically Sound

Relative strength → cross-sectional alpha

Momentum → time-series persistence

Vol scaling → variance control

Sortino → downside risk aware

Weekly frequency → low noise, low churn