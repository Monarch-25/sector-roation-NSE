import argparse
from dataclasses import dataclass
from typing import Iterable

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class BacktestConfig:
    max_allocation: float = 0.25
    min_allocation: float = 0.02
    rs_span: int = 100
    rs_center_window: int = 100
    mom_roc_days: int = 10
    mom_smooth_span: int = 10
    rebalance_weekday: int = 4


def _double_ema(s: pd.Series, span: int) -> pd.Series:
    e1 = s.ewm(span=span, adjust=False, min_periods=span).mean()
    return e1.ewm(span=span, adjust=False, min_periods=span).mean()


def load_prices(csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path, parse_dates=["Date"])
    df = df.set_index("Date").sort_index()
    df = df.apply(pd.to_numeric, errors="coerce")
    df = df.ffill()
    df = df.dropna(how="all")
    return df


def build_custom_benchmark(prices: pd.DataFrame, sector_cols: Iterable[str]) -> pd.Series:
    sector_rets = prices.loc[:, list(sector_cols)].pct_change(fill_method=None)
    mean_ret = sector_rets.mean(axis=1, skipna=True).fillna(0.0)
    bench = 100.0 * (1.0 + mean_ret).cumprod()
    bench.name = "Custom_Benchmark"
    return bench


def compute_rrg(prices: pd.DataFrame, benchmark_price: pd.Series, cfg: BacktestConfig) -> tuple[pd.DataFrame, pd.DataFrame]:
    aligned = prices.join(benchmark_price, how="inner")
    benchmark = aligned[benchmark_price.name]

    rs_ratio = {}
    rs_mom = {}

    for col in prices.columns:
        s = aligned[col]
        ratio = s / benchmark
        ratio_sm = _double_ema(ratio, cfg.rs_span)
        center = ratio_sm.rolling(cfg.rs_center_window, min_periods=cfg.rs_center_window).mean()
        rs = 100.0 * (ratio_sm / center)
        rs = _double_ema(rs, max(5, cfg.rs_span // 10))

        roc = rs.pct_change(cfg.mom_roc_days, fill_method=None)
        roc_sm = _double_ema(roc, cfg.mom_smooth_span)
        mom = 100.0 * (1.0 + roc_sm)

        rs_ratio[col] = rs
        rs_mom[col] = mom

    rs_ratio_df = pd.DataFrame(rs_ratio, index=aligned.index)
    rs_mom_df = pd.DataFrame(rs_mom, index=aligned.index)
    return rs_ratio_df, rs_mom_df


def _apply_min_and_caps(raw_w: pd.Series, max_alloc: float, min_alloc: float) -> pd.Series:
    w = raw_w.copy().astype(float)
    w[w < 0] = 0.0

    if w.sum() <= 0:
        return w * 0.0

    w = w / w.sum()

    while True:
        below = (w > 0) & (w < min_alloc)
        if not bool(below.any()):
            break
        w[below] = 0.0
        if w.sum() <= 0:
            return w * 0.0
        w = w / w.sum()

    capped = pd.Series(False, index=w.index)
    for _ in range(len(w) + 2):
        over = w > max_alloc
        if not bool(over.any()):
            break
        excess = (w[over] - max_alloc).sum()
        w[over] = max_alloc
        capped[over] = True

        under = (w > 0) & (~capped)
        if not bool(under.any()):
            break
        w[under] = w[under] + excess * (w[under] / w[under].sum())

    return w


def score_weights_for_date(rs_ratio: pd.DataFrame, rs_mom: pd.DataFrame, asof_date: pd.Timestamp, cfg: BacktestConfig) -> pd.Series:
    r = rs_ratio.loc[asof_date]
    m = rs_mom.loc[asof_date]

    leading = (r > 100.0) & (m > 100.0)
    improving = (r < 100.0) & (m > 100.0)

    eligible = leading | improving
    score = (r + m).where(eligible, other=0.0)

    w = _apply_min_and_caps(score, cfg.max_allocation, cfg.min_allocation)
    w.name = asof_date
    return w


def build_weight_schedule(rs_ratio: pd.DataFrame, rs_mom: pd.DataFrame, cfg: BacktestConfig) -> pd.DataFrame:
    idx = rs_ratio.index
    fridays = idx[idx.weekday == cfg.rebalance_weekday]

    weights = pd.DataFrame(0.0, index=idx, columns=rs_ratio.columns)

    for f in fridays:
        next_pos = idx.searchsorted(f + pd.Timedelta(days=1), side="left")
        if next_pos >= len(idx):
            continue
        start = idx[next_pos]

        w = score_weights_for_date(rs_ratio, rs_mom, f, cfg)

        nxt_fr = fridays[fridays > f]
        if len(nxt_fr) > 0:
            nf = nxt_fr[0]
            nxt_pos = idx.searchsorted(nf + pd.Timedelta(days=1), side="left")
            end = idx[nxt_pos] if nxt_pos < len(idx) else idx[-1] + pd.Timedelta(days=1)
        else:
            end = idx[-1] + pd.Timedelta(days=1)

        mask = (idx >= start) & (idx < end)
        weights.loc[mask, :] = w.values

    return weights


def _max_drawdown(equity: pd.Series) -> float:
    rolling_max = equity.cummax()
    dd = (equity / rolling_max) - 1.0
    return float(dd.min())


def _sharpe(daily_returns: pd.Series, annualization: int = 252) -> float:
    r = daily_returns.dropna()
    if len(r) < 2:
        return float("nan")
    vol = r.std(ddof=1)
    if vol == 0 or np.isnan(vol):
        return float("nan")
    return float((r.mean() / vol) * np.sqrt(annualization))


def _annualized_alpha(strat_ret: pd.Series, bench_ret: pd.Series, annualization: int = 252) -> float:
    df = pd.concat([strat_ret, bench_ret], axis=1, join="inner").dropna()
    if len(df) < 30:
        return float("nan")

    y = df.iloc[:, 0].to_numpy(dtype=float)
    x = df.iloc[:, 1].to_numpy(dtype=float)
    X = np.column_stack([np.ones_like(x), x])
    coef, _, _, _ = np.linalg.lstsq(X, y, rcond=None)
    alpha_daily = float(coef[0])
    return alpha_daily * annualization


def run_backtest(csv_path: str, cfg: BacktestConfig) -> dict:
    prices = load_prices(csv_path)

    sector_cols = [c for c in prices.columns if c != "Benchmark"]
    custom_bench = build_custom_benchmark(prices, sector_cols)

    sector_prices = prices.loc[:, sector_cols]
    sector_prices = sector_prices.loc[custom_bench.index]

    rs_ratio, rs_mom = compute_rrg(sector_prices, custom_bench, cfg)
    weights = build_weight_schedule(rs_ratio, rs_mom, cfg)

    sector_ret = sector_prices.pct_change(fill_method=None).fillna(0.0)
    strat_ret = (weights * sector_ret).sum(axis=1)

    bench_ret = custom_bench.pct_change(fill_method=None).fillna(0.0)

    strat_equity = 100.0 * (1.0 + strat_ret).cumprod()
    bench_equity = 100.0 * (1.0 + bench_ret).cumprod()

    results = {
        "prices": prices,
        "sector_prices": sector_prices,
        "custom_benchmark": custom_bench,
        "weights": weights,
        "strategy_returns": strat_ret,
        "benchmark_returns": bench_ret,
        "strategy_equity": strat_equity,
        "benchmark_equity": bench_equity,
    }
    return results


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--csv", default="data/processed/indian_sector_data_2013_2025.csv")
    p.add_argument("--max-allocation", type=float, default=0.25)
    p.add_argument("--min-allocation", type=float, default=0.02)
    args = p.parse_args()

    cfg = BacktestConfig(max_allocation=args.max_allocation, min_allocation=args.min_allocation)
    res = run_backtest(args.csv, cfg)

    sharpe = _sharpe(res["strategy_returns"])
    mdd = _max_drawdown(res["strategy_equity"])
    alpha = _annualized_alpha(res["strategy_returns"], res["benchmark_returns"])

    print("Strategy vs Custom_Benchmark")
    print(f"Sharpe Ratio: {sharpe:.3f}")
    print(f"Maximum Drawdown: {mdd:.3%}")
    print(f"Annualized Alpha: {alpha:.3%}")

    latest_date = res["weights"].index[-1]
    latest_w = res["weights"].loc[latest_date].sort_values(ascending=False)
    latest_w = latest_w[latest_w > 0]

    print("\nFinal Sector Weights (Most Recent Date)")
    if len(latest_w) == 0:
        print("No active positions")
    else:
        out = latest_w.to_frame("weight")
        out.index.name = "sector"
        print(out.to_string(float_format=lambda x: f"{x:.4f}"))


if __name__ == "__main__":
    main()
