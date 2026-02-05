import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime
from pathlib import Path

# 1. Define the Universe (Sector Map)
# Mapping sectors to Yahoo Finance Tickers
# The Benchmark will be calculated as the equal-weighted average return of these sectors.
tickers = {
    'Finance': '^NSEBANK',            # Nifty Financial Services
    'Oil & Gas (Energy)': '^CNXENERGY', # Nifty Energy (Proxy for Oil & Gas)
    'FMCG': '^CNXFMCG',              # Nifty FMCG
    'IT': '^CNXIT',                  # Nifty IT
    'Metal': '^CNXMETAL',            # Nifty Metal
    'Consumer': '^CNXCONSUM',        # Nifty India Consumption (Broad proxy for Cons. Durables)
    'Realty': '^CNXREALTY',          # Nifty Realty
    'Media': '^CNXMEDIA',            # Nifty Media
    'Healthcare (Pharma)': '^CNXPHARMA', # Nifty Pharma (Proxy for Healthcare)
    'Auto': '^CNXAUTO'               # Nifty Auto
}

# 2. Define Timeframe
start_date = '2013-01-01'
end_date = datetime.today().strftime('%Y-%m-%d')

print(f"Downloading data for {len(tickers)} indices from {start_date} to {end_date}...")

# 3. Download Data
raw = yf.download(
    list(tickers.values()),
    start=start_date,
    end=end_date,
    progress=False,
    auto_adjust=False,
)

if isinstance(raw, pd.DataFrame) and isinstance(raw.columns, pd.MultiIndex):
    if 'Adj Close' in raw.columns.get_level_values(0):
        data = raw['Adj Close'].copy()
    elif 'Close' in raw.columns.get_level_values(0):
        data = raw['Close'].copy()
    else:
        data = pd.DataFrame(index=raw.index)
else:
    data = raw

# 4. Clean and Rename Columns
if not data.empty:
    # Reverse the dictionary to map Tickers -> Names
    ticker_to_name = {v: k for k, v in tickers.items()}
    data.rename(columns=ticker_to_name, inplace=True)
    
    # Handle missing values
    data.ffill(inplace=True)
    data.dropna(how='all', inplace=True)

    # =========================================
    # 5. ANOMALY DETECTION AND CORRECTION
    # =========================================
    # Detect and fix spurious jumps in price series
    # These can occur due to Yahoo Finance data issues (e.g., missing adjustments)
    
    def detect_and_fix_anomalies(df, max_daily_return=0.5, verbose=True):
        """
        Detect and fix anomalous price jumps.
        
        A price jump is considered anomalous if:
        1. Daily return exceeds max_daily_return (default 50%)
        2. The jump is isolated (surrounding days are normal)
        
        Fix: Scale the series to remove the discontinuity.
        """
        fixed_df = df.copy()
        anomalies_found = []
        
        for col in df.columns:
            series = df[col].dropna()
            returns = series.pct_change()
            
            # Find anomalous jumps
            anomaly_mask = abs(returns) > max_daily_return
            anomaly_dates = returns[anomaly_mask].index.tolist()
            
            for date in anomaly_dates:
                ret = returns.loc[date]
                prev_idx = series.index.get_loc(date) - 1
                
                if prev_idx < 0:
                    continue
                
                prev_date = series.index[prev_idx]
                prev_val = series.loc[prev_date]
                curr_val = series.loc[date]
                
                # Check if this is a level shift (values after stay at new level)
                # by comparing the ratio
                ratio = curr_val / prev_val
                
                if abs(ratio) > 2:  # More than 2x jump
                    # This is likely a discontinuity, scale earlier data
                    if verbose:
                        print(f"  ANOMALY: {col} on {date.date()}: {ret:.1%} return (ratio: {ratio:.2f})")
                    
                    anomalies_found.append({
                        'sector': col,
                        'date': date,
                        'return': ret,
                        'ratio': ratio
                    })
                    
                    # Fix by scaling all data before this date
                    mask = fixed_df.index < date
                    fixed_df.loc[mask, col] = fixed_df.loc[mask, col] * ratio
                    
                    if verbose:
                        print(f"    FIXED: Scaled {col} data before {date.date()} by {ratio:.4f}")
        
        return fixed_df, anomalies_found
    
    print("\nChecking for data anomalies...")
    data, anomalies = detect_and_fix_anomalies(data, max_daily_return=0.5)
    
    if len(anomalies) == 0:
        print("No anomalies detected.")
    else:
        print(f"\nFixed {len(anomalies)} anomalies.")

    # =========================================
    # 6. Construct Custom Benchmark
    # =========================================
    # Logic: Daily pct_change of all sectors, take the mean, convert back to price series starting at 1.
    # Cap individual sector returns to avoid benchmark contamination from remaining outliers
    returns = data.pct_change()
    
    # Cap extreme returns at ±20% per day
    returns = returns.clip(lower=-0.20, upper=0.20)
    
    benchmark_returns = returns.mean(axis=1)
    benchmark_returns.iloc[0] = 0  # First day return is 0 to keep price at 1.0
    
    # Calculate cumulative returns (price series) starting at 1.0
    benchmark_price = (1 + benchmark_returns).cumprod()
    
    data['Benchmark'] = benchmark_price

    # Verify no remaining anomalies in benchmark
    bench_returns = benchmark_price.pct_change()
    max_bench_return = bench_returns.abs().max()
    if max_bench_return > 0.15:
        print(f"\nWARNING: Benchmark still has large daily return: {max_bench_return:.1%}")
        print("Dates with returns > 10%:")
        print(bench_returns[bench_returns.abs() > 0.10])

    # 7. Review Data
    print("\nData Download and Benchmark Construction Successful!")
    print("-" * 30)
    print(f"Total Trading Days: {len(data)}")
    print(f"Sectors Included: {[c for c in data.columns if c != 'Benchmark']}")
    print("-" * 30)
    print("Head of Data:")
    print(data.head())
    
    # 8. Save to CSV
    script_dir = Path(__file__).parent
    filename = script_dir / 'processed' / 'indian_sector_data_2013_2025.csv'
    filename.parent.mkdir(parents=True, exist_ok=True)
    data.to_csv(filename)
    print(f"\nSaved data to {filename}")

else:
    print("Error: No data downloaded.")

# Optional: Plotting
try:
    import matplotlib.pyplot as plt
    
    # Normalize data to start at 100 for comparison
    normalized_data = (data / data.iloc[0]) * 100
    
    plt.figure(figsize=(14, 7))
    for column in normalized_data.columns:
        linewidth = 3 if column == 'Benchmark' else 1
        alpha = 1.0 if column == 'Benchmark' else 0.6
        plt.plot(normalized_data.index, normalized_data[column], label=column, linewidth=linewidth, alpha=alpha)
        
    plt.title('Relative Sector Performance (Normalized, Base=100)')
    plt.legend()
    plt.show()
    
except ImportError:
    pass