import yfinance as yf
import pandas as pd
from datetime import datetime

# Define ticker symbol and date range
ticker_symbol = "AMZN"
start_date = "2015-01-01"
end_date = "2024-12-31"

# Print information about what we're fetching
print(f"Fetching daily stock data for {ticker_symbol} from {start_date} to {end_date}")

# Fetch the data
stock_data = yf.download(ticker_symbol, start=start_date, end=end_date, interval="1d")

# Display basic information about the data
print("\nData Overview:")
print(f"Shape: {stock_data.shape}")
print(f"Date Range: {stock_data.index.min()} to {stock_data.index.max()}")
print("\nFirst 5 rows:")
print(stock_data.head())
print("\nLast 5 rows:")
print(stock_data.tail())

# Save the data to a CSV filesv_filename = f"{ticker_symbol}_daily_data_{start_date}_to_{end_date.replace('-', '_')}.csv"
csv_path = f"/home/dnfy/Desktop/QuantInsti/data/{ticker_symbol}"
stock_data.to_csv(csv_path)
print(f"\nData saved to: {csv_path}")

# Calculate basic statistics
print("\nBasic Statistics:")
print(stock_data["Close"].describe())

# Check for missing values
missing_values = stock_data.isnull().sum()
if missing_values.sum() > 0:
    print("\nMissing Values:")
    print(missing_values)
else:
    print("\nNo missing values found in the data.")
