

"""
Data Download Module

This module provides utilities for downloading financial market data from Yahoo Finance.
It handles both individual stock ticker data and external market feature data (indices,
commodities, volatility measures, etc.) that can be used for machine learning model training.
"""

import os
import numpy as np
import pandas as pd
import yfinance as yf


def download_stock_data(ticker: str, start_date: str, end_date: str) -> pd.DataFrame:
    """
    Download adjusted close price and volume data for a given ticker.
    
    Args:
        ticker: Stock ticker symbol (e.g., 'AAPL', 'QQQ')
        start_date: Start date in format 'YYYY-MM-DD'
        end_date: End date in format 'YYYY-MM-DD'
    
    Returns:
        DataFrame with columns ['price', 'volume'] indexed by date
    
    Raises:
        ValueError: If no data is found for the ticker
    """
    # Download raw data from Yahoo Finance
    data = yf.download(
        ticker,
        start=start_date,
        end=end_date,
        auto_adjust=True,  # Automatically adjust OHLC for stock splits/dividends
        progress=False,    # Suppress progress bar
    )
    
    # Validate that data was successfully retrieved
    if data.empty:
        raise ValueError(f"No data found for ticker: {ticker}")
    
    # Handle MultiIndex columns (occurs when downloading multiple tickers or time periods)
    if isinstance(data.columns, pd.MultiIndex):
        # Extract first column if DataFrame, otherwise use Series as-is
        price = data["Close"].iloc[:, 0] if isinstance(data["Close"], pd.DataFrame) else data["Close"]
        volume = data["Volume"].iloc[:, 0] if isinstance(data["Volume"], pd.DataFrame) else data["Volume"]
    else:
        # Single index columns - direct access
        price = data["Close"]
        volume = data["Volume"]
    
    # Create clean DataFrame with renamed columns
    df = pd.DataFrame({"price": price, "volume": volume})
    # Remove any rows with missing data
    df.dropna(inplace=True)
    
    return df


def download_external_market_features(start_date: str, end_date: str) -> pd.DataFrame:
    """
    Download external market and macroeconomic features for model training.
    
    Features include:
    - Volatility: VIX (equity volatility)
    - Broad indices: SPY (S&P 500), IWM (Russell 2000 small-cap)
    - Sector ETFs: XLK (tech), XLF (finance), XLI (industrial), etc.
    - Asset classes: TLT (long-term bonds), HYG (high-yield bonds), GLD (gold)
    - Forex: UUP (US Dollar Index)
    - Rates: TNX (10-year Treasury yield)
    
    Args:
        start_date: Start date in format 'YYYY-MM-DD'
        end_date: End date in format 'YYYY-MM-DD'
    
    Returns:
        DataFrame with close prices and log returns for each feature, plus derived metrics
    """
    # Mapping of descriptive names to Yahoo Finance ticker symbols
    tickers = {
        "vix": "^VIX", "spy": "SPY", "iwm": "IWM", "xlk": "XLK",
        "xlf": "XLF", "xli": "XLI", "xly": "XLY", "xlp": "XLP",
        "tlt": "TLT", "hyg": "HYG", "uup": "UUP", "gld": "GLD", "tnx": "^TNX",
    }
    
    feature_data = pd.DataFrame()
    
    # Download each market feature and extract close prices and returns
    for name, ticker in tickers.items():
        # Fetch data from Yahoo Finance
        data = yf.download(ticker, start=start_date, end=end_date, auto_adjust=True, progress=False)
        if data.empty:
            print(f"Warning: No data for {ticker}")
            continue
        
        # Extract close price, handling MultiIndex columns if present
        close = data["Close"].iloc[:, 0] if isinstance(data["Close"], pd.DataFrame) else data["Close"]
        
        # Store close price and calculate log returns (price changes)
        feature_data[f"{name}_close"] = close
        feature_data[f"{name}_return"] = np.log(close).diff()
    
    # Remove rows with any missing values
    feature_data.dropna(inplace=True)
    
    # Create derived technical features that capture market relationships
    # These ratios capture relative strength between market segments
    feature_data["iwm_spy_relative_strength"] = feature_data["iwm_close"] / feature_data["spy_close"]
    feature_data["qqq_spy_market_strength"] = feature_data["xlk_close"] / feature_data["spy_close"]
    feature_data["tech_vs_staples"] = feature_data["xlk_close"] / feature_data["xlp_close"]
   
    
    return feature_data
