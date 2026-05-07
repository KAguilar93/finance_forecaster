

import os
import pandas as pd
import yfinance as yf

def download_stock_data(ticker: str, start_date: str, end_date: str) -> pd.DataFrame:
    """Download adjusted close price and volume data."""
    data = yf.download(
        ticker,
        start=start_date,
        end=end_date,
        auto_adjust=True,
        progress=False,
    )
    
    if data.empty:
        raise ValueError(f"No data found for ticker: {ticker}")
    
    if isinstance(data.columns, pd.MultiIndex):
        price = data["Close"].iloc[:, 0] if isinstance(data["Close"], pd.DataFrame) else data["Close"]
        volume = data["Volume"].iloc[:, 0] if isinstance(data["Volume"], pd.DataFrame) else data["Volume"]
    else:
        price = data["Close"]
        volume = data["Volume"]
    
    df = pd.DataFrame({"price": price, "volume": volume})
    df.dropna(inplace=True)
    
    return df


def download_external_market_features(start_date: str, end_date: str) -> pd.DataFrame:
    """Download external market and macro features."""
    tickers = {
        "vix": "^VIX", "spy": "SPY", "iwm": "IWM", "xlk": "XLK",
        "xlf": "XLF", "xli": "XLI", "xly": "XLY", "xlp": "XLP",
        "tlt": "TLT", "hyg": "HYG", "uup": "UUP", "gld": "GLD", "tnx": "^TNX",
    }
    
    feature_data = pd.DataFrame()
    
    for name, ticker in tickers.items():
        data = yf.download(ticker, start=start_date, end=end_date, auto_adjust=True, progress=False)
        if data.empty:
            print(f"Warning: No data for {ticker}")
            continue
            
        close = data["Close"].iloc[:, 0] if isinstance(data["Close"], pd.DataFrame) else data["Close"]
        feature_data[f"{name}_close"] = close
        feature_data[f"{name}_return"] = np.log(close).diff()
    
    feature_data.dropna(inplace=True)
    
    # Add derived features
    feature_data["iwm_spy_relative_strength"] = feature_data["iwm_close"] / feature_data["spy_close"]
    feature_data["qqq_spy_market_strength"] = feature_data["xlk_close"] / feature_data["spy_close"]
    feature_data["tech_vs_staples"] = feature_data["xlk_close"] / feature_data["xlp_close"]
    # ... (add the rest of your derived features here - I'll help with the full version next if needed)
    
    return feature_data
