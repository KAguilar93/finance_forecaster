"""Data Pipeline for QQQ Financial Forecasting with DVC Integration

This module implements a complete data pipeline:
- Downloads QQQ stock prices and external market features from Yahoo Finance
- Engineers relative strength and market regime indicators
- Integrates with DVC (Data Version Control) for Google Drive backup/sync
- Merges and processes data for model training

DVC Workflow:
1. Pull existing data from Google Drive (fast cache)
2. Download missing data from Yahoo Finance (if needed)
3. Process and merge into training datasets

Output files:
- data/raw/qqq_raw.csv: Raw QQQ prices and volume
- data/raw/market_features.csv: Raw external market features
- data/processed/qqq_processed.csv: Merged QQQ + features with log returns
- data/processed/market_features_processed.csv: Processed market features copy
"""

import logging
import subprocess
from pathlib import Path
import pandas as pd
import numpy as np
import yfinance as yf
import subprocess

logger = logging.getLogger(__name__)

# Directory paths for raw backups and processed data
RAW_DIR = Path("data/raw")
PROCESSED_DIR = Path("data/processed")

import numpy as np
import pandas as pd
import yfinance as yf

logger = logging.getLogger(__name__)

# Directory paths for raw backups and processed data
RAW_DIR = Path("data/raw")
PROCESSED_DIR = Path("data/processed")


def download_stock_data(ticker: str = "QQQ", start: str = "2015-01-01", end: str = "2026-05-15") -> pd.DataFrame:
    """
    Download historical stock price and volume data from Yahoo Finance.

    Handles yfinance quirks:
    - MultiIndex columns for certain tickers
    - Automatic adjustment for stock splits and dividends

    Args:
        ticker (str): Stock ticker symbol (default: QQQ - Nasdaq-100 ETF)
        start (str): Start date in YYYY-MM-DD format
        end (str): End date in YYYY-MM-DD format

    Returns:
        pd.DataFrame: DataFrame with 'price' (close) and 'volume', indexed by date
    """
    logger.info(f"Downloading main ticker: {ticker}")
    # Download with auto_adjust to handle stock splits and dividends automatically
    data = yf.download(ticker, start=start, end=end, auto_adjust=True, progress=False)

    # Handle MultiIndex columns: yfinance returns MultiIndex (tuple format) for certain tickers
    if isinstance(data.columns, pd.MultiIndex):
        close = data.get(("Close", ticker), data["Close"])
        volume = data.get(("Volume", ticker), data["Volume"])
    else:
        # Simple case: single ticker gives simple column index
        close = data["Close"]
        volume = data["Volume"]

    # Create clean DataFrame with standardized column names
    df = pd.DataFrame({"price": close, "volume": volume})
    # Remove rows with missing data
    df.dropna(inplace=True)
    return df


def download_external_market_features(start: str = "2015-01-01", end: str = "2026-05-15") -> pd.DataFrame:
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
    - Market breadth: IWM/SPY (small vs large caps), XLK/SPY (tech strength)
    - Sector rotation: XLY/XLP (growth vs defensive), XLK/XLP (tech vs staples)
    - Risk indicators: XLF/SPY (financial health), XLI/SPY (industrial health)
    - Risk appetite: HYG/TLT (credit spreads), GLD/SPY (safe-haven demand)

    Args:
        start (str): Start date in YYYY-MM-DD format
        end (str): End date in YYYY-MM-DD format

    Returns:
        pd.DataFrame: DataFrame with close prices, log returns, and engineered features
    """
    logger.info("Downloading external market features...")
    # Mapping of descriptive names to Yahoo Finance tickers
    tickers = {
        "vix": "^VIX",
        "spy": "SPY",
        "iwm": "IWM",
        "xlk": "XLK",
        "xlf": "XLF",
        "xli": "XLI",
        "xly": "XLY",
        "xlp": "XLP",
        "tlt": "TLT",
        "hyg": "HYG",
        "uup": "UUP",
        "gld": "GLD",
        "tnx": "^TNX",
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

    # Remove rows with any missing values before feature engineering
    feature_data.dropna(inplace=True)

    # === FEATURE ENGINEERING: Relative Strength Ratios ===
    # These indicators capture market dynamics and risk regimes
    # Ratio > 1 indicates strength in numerator asset class

    # Market breadth: small caps (IWM) vs large caps (SPY)
    # Rising ratio: risk appetite, falling ratio: risk-off
    feature_data["iwm_spy_relative_strength"] = feature_data["iwm_close"] / feature_data["spy_close"]

    # Tech sector (XLK) vs broad market (SPY) - growth indicator
    feature_data["qqq_spy_market_strength"] = feature_data["xlk_close"] / feature_data["spy_close"]

    # Tech (XLK) vs Consumer Staples (XLP): growth vs defensive rotation
    # Rising: growth environment, Falling: defensive/recessionary
    feature_data["tech_vs_staples"] = feature_data["xlk_close"] / feature_data["xlp_close"]

    # Consumer Discretionary vs Staples: economic optimism indicator
    # Rising: optimism, Falling: pessimism or slowdown
    feature_data["discretionary_vs_staples"] = feature_data["xly_close"] / feature_data["xlp_close"]

    # Financials sector strength relative to market
    # Indicator of banking system health and lending conditions
    feature_data["financials_vs_market"] = feature_data["xlf_close"] / feature_data["spy_close"]

    # Industrials sector strength (economic health indicator)
    # Rising: strong economic growth, Falling: slowdown
    feature_data["industrials_vs_market"] = feature_data["xli_close"] / feature_data["spy_close"]

    # Credit risk appetite: high-yield bonds vs safe Treasuries (spread proxy)
    # Rising: investors take risk, Falling: flight to safety
    feature_data["credit_risk_appetite"] = feature_data["hyg_close"] / feature_data["tlt_close"]

    # Gold strength: safe-haven demand indicator
    # Rising: risk-off environment, Falling: risk-on environment
    feature_data["gold_vs_market"] = feature_data["gld_close"] / feature_data["spy_close"]

    return feature_data


def process_data() -> None:
    """
    Main data processing pipeline with DVC integration.

    Workflow:
    1. Create directories (data/raw, data/processed)
    2. Attempt to pull data from Google Drive via DVC (cache optimization)
    3. Download missing data from Yahoo Finance (fallback)
    4. Merge datasets and create final training files

    Files created:
    - qqq_raw.csv: Cache of QQQ prices (never overwritten)
    - market_features.csv: Cache of market indicators (never overwritten)
    - qqq_processed.csv: Merged dataset with engineered features + log returns
    - market_features_processed.csv: Copy of market features for reference
    """
    # Create output directories if they don't exist
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    # === DVC PULL: Retrieve cached data from Google Drive ===
    # This is fast since data is already processed and stored remotely
    logger.info("Checking Google Drive for data...")
    try:
        # Quietly pull data from DVC remote (Google Drive)
        subprocess.run(["dvc", "pull", "--quiet"], check=True)
        logger.info("Data pulled from Google Drive")
    except Exception:
        # DVC pull failed or no data available; will download fresh data
        logger.info("No data on Google Drive. Downloading fresh...")

    # === DOWNLOAD MISSING DATA ===
    # Download QQQ if not already cached (first run or cache miss)
    if not (RAW_DIR / "qqq_raw.csv").exists():
        df = download_stock_data()
        df.to_csv(RAW_DIR / "qqq_raw.csv")
        logger.info("Downloaded qqq_raw.csv")

    # Download market features if not already cached
    if not (RAW_DIR / "market_features.csv").exists():
        market_df = download_external_market_features()
        market_df.to_csv(RAW_DIR / "market_features.csv")
        logger.info("Downloaded market_features.csv")

    # === CREATE PROCESSED MAIN DATASET ===
    # Merge raw QQQ with market features, engineer final training dataset
    if not (PROCESSED_DIR / "qqq_processed.csv").exists():
        # Load raw data files
        qqq = pd.read_csv(RAW_DIR / "qqq_raw.csv", index_col=0, parse_dates=True)
        market = pd.read_csv(RAW_DIR / "market_features.csv", index_col=0, parse_dates=True)

        # Left join: preserve all QQQ dates; forward fill market features for any gaps
        df = qqq.join(market, how="left").ffill()

        # Create target variable: log returns (used by GARCH, LSTM, backtesting)
        # log_return = log(price_t) - log(price_t-1)
        df["log_return"] = np.log(df["price"]).diff()

        # Remove rows with NaN (from join operations and diff)
        df.dropna(inplace=True)

        # Save final processed dataset for model training
        df.to_csv(PROCESSED_DIR / "qqq_processed.csv")
        logger.info(f"Created qqq_processed.csv ({df.shape[0]} rows)")

    # === CREATE MARKET FEATURES REFERENCE FILE ===
    # Save processed version of market features for analysis/reference
    if not (PROCESSED_DIR / "market_features_processed.csv").exists():
        # Load market features from raw
        market = pd.read_csv(RAW_DIR / "market_features.csv", index_col=0, parse_dates=True)
        # Save to processed folder for easy reference
        market.to_csv(PROCESSED_DIR / "market_features_processed.csv")
        logger.info("Created market_features_processed.csv")

    logger.info("Data pipeline complete")


def main() -> None:
    """
    Entry point for the data pipeline.
    Configures logging and executes the data processing workflow.
    """
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    process_data()


if __name__ == "__main__":
    # Run pipeline when file is executed directly
    main()
