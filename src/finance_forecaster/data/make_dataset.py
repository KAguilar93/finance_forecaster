"""Data Pipeline for QQQ Financial Forecasting with DVC Integration

This module implements a complete data pipeline:
- Downloads QQQ stock prices and external market features from Yahoo Finance
- Engineers relative strength and market regime indicators
- Integrates with DVC (Data Version Control) for Google Drive backup/sync
- Merges and processes data for model training

DVC Workflow:
1. Pull existing data from Google Drive (fast, cached)
2. Download only missing data from Yahoo Finance (if needed)
3. Process and merge into training dataset
4. Push new data to Google Drive if downloaded
"""

import logging
from pathlib import Path
import pandas as pd
import numpy as np
import yfinance as yf
import subprocess

logger = logging.getLogger(__name__)

# Directory paths for raw backups and processed data
RAW_DIR = Path("data/raw")
PROCESSED_DIR = Path("data/processed")

def download_stock_data(ticker="QQQ", start="2015-01-01", end="2026-05-15"):
    """
    Download historical stock price and volume data from Yahoo Finance.
    
    Handles yfinance quirks:
    - MultiIndex columns when downloading certain tickers
    - Automatic adjustment for stock splits and dividends
    
    Args:
        ticker (str): Stock ticker symbol (default: QQQ - Nasdaq-100 ETF)
        start (str): Start date in YYYY-MM-DD format
        end (str): End date in YYYY-MM-DD format
    
    Returns:
        pd.DataFrame: DataFrame with columns 'price' and 'volume', indexed by date
    """
    logger.info(f"Downloading main ticker: {ticker}")
    # Download with auto_adjust to handle stock splits and dividends automatically
    data = yf.download(ticker, start=start, end=end, auto_adjust=True, progress=False)
    
    # Handle MultiIndex columns: yfinance returns MultiIndex (tuple format) for certain tickers
    if isinstance(data.columns, pd.MultiIndex):
        close = data.get(('Close', ticker), data['Close'])
        volume = data.get(('Volume', ticker), data['Volume'])
    else:
        # Simple case: single ticker gives simple column index
        close = data['Close']
        volume = data['Volume']
    
    # Create clean dataframe with standardized column names
    df = pd.DataFrame({"price": close, "volume": volume})
    # Remove any rows with missing data
    df.dropna(inplace=True)
    return df


def download_external_market_features(start="2015-01-01", end="2026-05-15"):
    """
    Download external market features and compute relative strength indicators.
    
    Features downloaded:
    - Volatility: VIX (fear index), TNX (10Y yields)
    - Equities: SPY (S&P 500), IWM (Russell 2000 small caps)
    - Sectors: XLK (tech), XLF (financials), XLI (industrials), XLY (discretionary), XLP (staples)
    - Fixed Income: TLT (long-term bonds), HYG (high-yield credit)
    - Currency: UUP (USD index)
    - Commodities: GLD (gold)
    
    Engineered features (relative strength ratios):
    - Market breadth: IWM/SPY, XLK/SPY, XLI/SPY
    - Sector rotation: XLY/XLP, XLK/XLP, XLF/SPY
    - Risk appetite: HYG/TLT (credit spreads), GLD/SPY
    
    Args:
        start (str): Start date in YYYY-MM-DD format
        end (str): End date in YYYY-MM-DD format
    
    Returns:
        pd.DataFrame: DataFrame with close prices, log returns, and engineered features
    """
    logger.info("Downloading external market features...")
    # Mapping of descriptive names to Yahoo Finance tickers
    tickers = {
        "vix": "^VIX", "spy": "SPY", "iwm": "IWM", "xlk": "XLK",
        "xlf": "XLF", "xli": "XLI", "xly": "XLY", "xlp": "XLP",
        "tlt": "TLT", "hyg": "HYG", "uup": "UUP", "gld": "GLD", "tnx": "^TNX",
    }
    
    feature_data = pd.DataFrame()
    
    # Download each feature and store closing price and log returns
    for name, ticker in tickers.items():
        # Download data with auto_adjust for split/dividend handling
        data = yf.download(ticker, start=start, end=end, auto_adjust=True, progress=False)
        if data.empty:
            # Skip tickers with no data (network issues, invalid tickers)
            continue
        
        # Handle MultiIndex column format from yfinance
        # Take first column if Close is a DataFrame (multiple tickers downloaded)
        close = data["Close"].iloc[:, 0] if isinstance(data["Close"], pd.DataFrame) else data["Close"]
        
        # Store closing price and compute log returns (price momentum indicator)
        feature_data[f"{name}_close"] = close
        feature_data[f"{name}_return"] = np.log(close).diff()
    
    # Remove rows with any missing values
    feature_data.dropna(inplace=True)
    
    # === FEATURE ENGINEERING: Relative Strength Ratios ===
    # These indicators capture market dynamics and risk regimes
    # Ratio > 1 indicates strength in numerator asset class
    
    # Market breadth: small caps (IWM) vs large caps (SPY)
    feature_data["iwm_spy_relative_strength"] = feature_data["iwm_close"] / feature_data["spy_close"]
    # Tech sector strength
    feature_data["qqq_spy_market_strength"] = feature_data["xlk_close"] / feature_data["spy_close"]
    # Tech (XLK) vs Consumer Staples (XLP): growth vs defensive
    feature_data["tech_vs_staples"] = feature_data["xlk_close"] / feature_data["xlp_close"]
    # Consumer Discretionary vs Staples: economic optimism indicator
    feature_data["discretionary_vs_staples"] = feature_data["xly_close"] / feature_data["xlp_close"]
    # Financials sector strength relative to market
    feature_data["financials_vs_market"] = feature_data["xlf_close"] / feature_data["spy_close"]
    # Industrials sector strength (economic health indicator)
    feature_data["industrials_vs_market"] = feature_data["xli_close"] / feature_data["spy_close"]
    # Credit risk appetite: high-yield bonds vs safe Treasuries (spread proxy)
    feature_data["credit_risk_appetite"] = feature_data["hyg_close"] / feature_data["tlt_close"]
    # Gold strength: safe-haven demand indicator
    feature_data["gold_vs_market"] = feature_data["gld_close"] / feature_data["spy_close"]
    
    return feature_data


def process_data():
    """
    Main data processing pipeline with DVC integration.
    
    Workflow:
    1. Create directories (data/raw, data/processed)
    2. Attempt to pull data from Google Drive via DVC (cache)
    3. Download missing data from Yahoo Finance (if needed)
    4. Merge datasets and create final training file
    5. Push new data to Google Drive if downloaded
    
    DVC Optimization:
    - Tracks whether new data was downloaded
    - Only pushes to Google Drive if new data was created
    - Speeds up subsequent runs by using cached data
    """
    # Create output directories
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    # Track if we downloaded new data (determines if we push to Google Drive)
    data_was_downloaded = False

    # === DVC PULL: Retrieve cached data from Google Drive ===
    # This is fast since data is already processed and stored
    logger.info("Checking Google Drive for data...")
    try:
        # Quietly pull data from DVC remote (Google Drive)
        subprocess.run(["dvc", "pull", "--quiet"], check=True)
        logger.info("Data pulled from Google Drive")
    except Exception:
        # No data available on Google Drive, will download fresh
        logger.info("No data on Google Drive. Downloading fresh...")
        data_was_downloaded = True

    # === DOWNLOAD MISSING DATA ===
    # Download QQQ if not already cached
    if not (RAW_DIR / "qqq_raw.csv").exists():
        df = download_stock_data()
        df.to_csv(RAW_DIR / "qqq_raw.csv")
        logger.info("Downloaded qqq_raw.csv")
        data_was_downloaded = True

    # Download market features if not already cached
    if not (RAW_DIR / "market_features.csv").exists():
        market_df = download_external_market_features()
        market_df.to_csv(RAW_DIR / "market_features.csv")
        logger.info("Downloaded market_features.csv")
        data_was_downloaded = True

    # === CREATE PROCESSED DATASET ===
    # Merge raw data and create final training file
    if not (PROCESSED_DIR / "qqq_processed.csv").exists():
        # Load raw data files
        qqq = pd.read_csv(RAW_DIR / "qqq_raw.csv", index_col=0, parse_dates=True)
        market = pd.read_csv(RAW_DIR / "market_features.csv", index_col=0, parse_dates=True)
        
        # Left join: preserve all QQQ dates; fill forward market features
        df = qqq.join(market, how="left").ffill()
        
        # Create target variable: log returns (used by GARCH, LSTM, evaluation)
        df["log_return"] = np.log(df["price"]).diff()
        # Remove rows with NaN (from join and diff operations)
        df.dropna(inplace=True)
        
        # Save final processed dataset
        df.to_csv(PROCESSED_DIR / "qqq_processed.csv")
        logger.info(f"Created processed file with {len(df)} rows")
        data_was_downloaded = True

    # === DVC PUSH: Save new data to Google Drive ===
    # Only push if we downloaded new data (optimization)
    if data_was_downloaded:
        try:
            logger.info("Pushing new data to Google Drive via DVC...")
            # Add data directories to DVC tracking
            subprocess.run(["dvc", "add", "data/raw", "data/processed"], check=True)
            # Push to remote (Google Drive)
            subprocess.run(["dvc", "push", "--quiet"], check=True)
            logger.info("Successfully pushed new data to Google Drive")
        except Exception as e:
            # DVC push failure (not fatal; data is still local)
            logger.warning(f"DVC push failed: {e}")
    else:
        # No new data downloaded; using existing cached data from Google Drive
        logger.info("No new data to push. Using existing data from Google Drive.")

    logger.info("Data pipeline complete")


def main():
    """
    Entry point for the data pipeline.
    Configures logging and executes the data processing workflow.
    """
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    process_data()


if __name__ == "__main__":
    # Run pipeline when file is executed directly
    main()
