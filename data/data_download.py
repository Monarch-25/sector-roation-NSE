import yfinance as yf
import pandas as pd
from datetime import datetime
from pathlib import Path

# 1. Define the Universe (Sector Map)
# Mapping your requested sectors to Yahoo Finance Tickers
# Note: We also add '^NSEI' (Nifty 50) as it is required as the Benchmark for RRG calculations.
tickers = {
    'Benchmark': '^NSEI',            # Nifty 50
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
# We download 'Adj Close' to account for any corporate actions/splits
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
# If data was downloaded successfully, we rename columns from Tickers to Readable Names
if not data.empty:
    # Reverse the dictionary to map Tickers -> Names for renaming
    ticker_to_name = {v: k for k, v in tickers.items()}
    data.rename(columns=ticker_to_name, inplace=True)
    
    # Handle missing values (forward fill first, then drop any remaining leading NaNs)
    data.ffill(inplace=True)
    if 'Benchmark' in data.columns:
        data.dropna(subset=['Benchmark'], inplace=True)
    else:
        data.dropna(how='all', inplace=True)

    # 5. Review Data
    print("\nData Download Successful!")
    print("-" * 30)
    print(f"Total Trading Days: {len(data)}")
    print(f"Sectors Included: {data.columns.tolist()}")
    print("-" * 30)
    print("Head of Data:")
    print(data.head())
    
    # 6. Save to CSV (Optional)
    filename = Path('processed') / 'indian_sector_data_2013_2025.csv'
    filename.parent.mkdir(parents=True, exist_ok=True)
    data.to_csv(filename)
    print(f"\nSaved data to {filename}")

else:
    print("Error: No data downloaded. Please check your internet connection or ticker symbols.")

# Optional: Plotting to verify relative performance (Normalized to 100)
# This is NOT the RRG, just a normalized price chart to check data quality
try:
    import matplotlib.pyplot as plt
    
    # Normalize data to start at 100
    normalized_data = (data / data.iloc[0]) * 100
    
    plt.figure(figsize=(14, 7))
    for column in normalized_data.columns:
        # Highlight Benchmark
        linewidth = 3 if column == 'Benchmark' else 1
        alpha = 1.0 if column == 'Benchmark' else 0.6
        plt.plot(normalized_data.index, normalized_data[column], label=column, linewidth=linewidth, alpha=alpha)
        
    plt.title('Relative Sector Performance (Normalized, 2013=100)')
    plt.legend()
    plt.show()
    
except ImportError:
    print("Matplotlib not found. Skipping plot.")