"""Data Pipeline for QQQ Financial Forecasting

This module handles the complete data pipeline:
- Downloads QQQ stock prices and volumes from Yahoo Finance
- Downloads external market features (VIX, sector ETFs, utilities, commodities)
- Engineers relative strength and market regime indicators
- Merges and processes data for model training
"""

import logging
from pathlib import Path
import pandas as pd
import numpy as np
import yfinance as yf

logger = logging.getLogger(__name__)

# Directory paths for raw backups and processed data
RAW_DIR = Path("data/raw")
PROCESSED_DIR = Path("data/processed")

def download_stock_data(ticker="QQQ", start="2015-01-01", end="2026-05-15"):
    """
    Download historical stock price and volume data from Yahoo Finance.
    
    Handles yfinance quirks:
    - MultiIndex columns when downloading multiple tickers
    - Empty data responses (network issues, invalid tickers)
    
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
    
    # Handle empty response (network error or invalid ticker)
    if data.empty:
        logger.error("No data returned from yfinance")
        return pd.DataFrame()
    
    # Handle MultiIndex columns: yfinance returns MultiIndex when downloading certain tickers
    # Check if columns are MultiIndex (tuple format) vs simple Index
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
            logger.warning(f"No data found for {ticker}")
            continue
        
        # Handle MultiIndex column format from yfinance
        if isinstance(data.columns, pd.MultiIndex):
            close = data["Close"]
            # If Close is a DataFrame (multiple tickers), take first column
            if isinstance(close, pd.DataFrame):
                close = close.iloc[:, 0]
        else:
            # Simple index: directly access Close column
            close = data["Close"]
        
        # Store closing price and compute log returns (price momentum indicator)
        feature_data[f"{name}_close"] = close
        feature_data[f"{name}_return"] = np.log(close).diff()
    
    # Remove rows with missing values before feature engineering
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
    
    logger.info(f"Created {feature_data.shape[1]} market feature columns")
    return feature_data


def process_data():
    """
    Main data processing pipeline:
    1. Creates necessary directories
    2. Downloads and caches raw data (QQQ prices, market features)
    3. Creates separate processed versions with feature engineering
    4. Merges datasets and creates target variables
    5. Saves final dataset for model training
    
    File organization:
    - data/raw/: Backup copies of downloaded data (never overwritten)
    - data/processed/: Analysis-ready data with engineered features
    """
    # Create directories if they don't exist
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    # === FILE PATHS ===
    # Raw backups: used as cache to avoid re-downloading
    qqq_raw = RAW_DIR / "qqq_raw.csv"
    market_raw = RAW_DIR / "market_features.csv"

    # Processed files: ready for model training
    qqq_processed = PROCESSED_DIR / "qqq_processed.csv"
    market_processed = PROCESSED_DIR / "market_features_processed.csv"

    # === DOWNLOAD RAW DATA (First Run Only) ===
    # Download QQQ if not cached
    if not qqq_raw.exists():
        df = download_stock_data()
        df.to_csv(qqq_raw)
        logger.info(f"Saved raw backup: {qqq_raw.name}")
    else:
        logger.info("Raw QQQ backup exists.")

    # Download market features if not cached
    if not market_raw.exists():
        market_df = download_external_market_features()
        market_df.to_csv(market_raw)
        logger.info(f"Saved raw backup: {market_raw.name}")
    else:
        logger.info("Raw market features backup exists.")

    # === SAVE MARKET FEATURES TO PROCESSED FOLDER ===
    # Creates a copy in processed folder for reference/debugging
    if not market_processed.exists():
        market_df = pd.read_csv(market_raw, index_col=0, parse_dates=True)
        market_df.to_csv(market_processed)
        logger.info(f"Saved market features to processed: {market_processed.name}")
    else:
        logger.info("Market features already in processed folder.")

    # === CREATE MAIN PROCESSED DATASET ===
    # Merge QQQ with market features and create final training dataset
    if not qqq_processed.exists():
        logger.info("Creating merged processed dataset...")
        # Load raw backups
        qqq = pd.read_csv(qqq_raw, index_col=0, parse_dates=True)
        market = pd.read_csv(market_raw, index_col=0, parse_dates=True)

        # Left join: preserve all QQQ dates; fill forward market features
        df = qqq.join(market, how="left")
        df.ffill(inplace=True)  # Forward fill missing values from market features

        # Create target variable: log returns (used by GARCH, LSTM, evaluation)
        df["log_return"] = np.log(df["price"]).diff()
        # Remove rows with NaN (from join and diff operations)
        df.dropna(inplace=True)

        # Save final processed dataset
        df.to_csv(qqq_processed)
        logger.info(f"Processed data saved → {qqq_processed.name} ({df.shape[0]} rows)")
    else:
        logger.info("Processed data already exists.")

    logger.info("Data pipeline complete")
    return True


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
