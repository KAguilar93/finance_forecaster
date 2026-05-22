import logging
from pathlib import Path
import pandas as pd
import numpy as np
import yfinance as yf
import subprocess

logger = logging.getLogger(__name__)

RAW_DIR = Path("data/raw")
PROCESSED_DIR = Path("data/processed")

def download_stock_data(ticker="QQQ", start="2015-01-01", end="2026-05-15"):
    logger.info(f"Downloading main ticker: {ticker}")
    data = yf.download(ticker, start=start, end=end, auto_adjust=True, progress=False)
    if isinstance(data.columns, pd.MultiIndex):
        close = data.get(('Close', ticker), data['Close'])
        volume = data.get(('Volume', ticker), data['Volume'])
    else:
        close = data['Close']
        volume = data['Volume']
    df = pd.DataFrame({"price": close, "volume": volume})
    df.dropna(inplace=True)
    return df


def download_external_market_features(start="2015-01-01", end="2026-05-15"):
    logger.info("Downloading external market features...")
    tickers = {
        "vix": "^VIX", "spy": "SPY", "iwm": "IWM", "xlk": "XLK",
        "xlf": "XLF", "xli": "XLI", "xly": "XLY", "xlp": "XLP",
        "tlt": "TLT", "hyg": "HYG", "uup": "UUP", "gld": "GLD", "tnx": "^TNX",
    }
    feature_data = pd.DataFrame()
    for name, ticker in tickers.items():
        data = yf.download(ticker, start=start, end=end, auto_adjust=True, progress=False)
        if data.empty:
            continue
        close = data["Close"].iloc[:, 0] if isinstance(data["Close"], pd.DataFrame) else data["Close"]
        feature_data[f"{name}_close"] = close
        feature_data[f"{name}_return"] = np.log(close).diff()
    
    feature_data.dropna(inplace=True)
    
    # Feature engineering
    feature_data["iwm_spy_relative_strength"] = feature_data["iwm_close"] / feature_data["spy_close"]
    feature_data["qqq_spy_market_strength"] = feature_data["xlk_close"] / feature_data["spy_close"]
    feature_data["tech_vs_staples"] = feature_data["xlk_close"] / feature_data["xlp_close"]
    feature_data["discretionary_vs_staples"] = feature_data["xly_close"] / feature_data["xlp_close"]
    feature_data["financials_vs_market"] = feature_data["xlf_close"] / feature_data["spy_close"]
    feature_data["industrials_vs_market"] = feature_data["xli_close"] / feature_data["spy_close"]
    feature_data["credit_risk_appetite"] = feature_data["hyg_close"] / feature_data["tlt_close"]
    feature_data["gold_vs_market"] = feature_data["gld_close"] / feature_data["spy_close"]
    
    return feature_data


def process_data():
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    logger.info("Checking Google Drive for data...")
    try:
        subprocess.run(["dvc", "pull", "--quiet"], check=True)
        logger.info("Data pulled from Google Drive")
    except Exception:
        logger.info("No data on Google Drive. Downloading fresh...")

    # Download raw files if missing
    if not (RAW_DIR / "qqq_raw.csv").exists():
        df = download_stock_data()
        df.to_csv(RAW_DIR / "qqq_raw.csv")
        logger.info("Downloaded qqq_raw.csv")

    if not (RAW_DIR / "market_features.csv").exists():
        market_df = download_external_market_features()
        market_df.to_csv(RAW_DIR / "market_features.csv")
        logger.info("Downloaded market_features.csv")

    # Create processed file with moving averages
    if not (PROCESSED_DIR / "qqq_processed.csv").exists():
        qqq = pd.read_csv(RAW_DIR / "qqq_raw.csv", index_col=0, parse_dates=True)
        market = pd.read_csv(RAW_DIR / "market_features.csv", index_col=0, parse_dates=True)

        df = qqq.join(market, how="left").ffill()
        df["log_return"] = np.log(df["price"]).diff()

        # Add moving averages needed for training
        df["ma_20"] = df["price"].rolling(20).mean()
        df["ma_50"] = df["price"].rolling(50).mean()
        df["volatility_ratio"] = df["log_return"].rolling(5).std() / df["log_return"].rolling(20).std()

        df.dropna(inplace=True)
        df.to_csv(PROCESSED_DIR / "qqq_processed.csv")
        logger.info(f"Created qqq_processed.csv with moving averages ({df.shape[0]} rows)")

    # Save processed market features
    if not (PROCESSED_DIR / "market_features_processed.csv").exists():
        market = pd.read_csv(RAW_DIR / "market_features.csv", index_col=0, parse_dates=True)
        market.to_csv(PROCESSED_DIR / "market_features_processed.csv")
        logger.info("Created market_features_processed.csv")

    logger.info("Data pipeline complete")


def main():
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    process_data()


if __name__ == "__main__":
    main()
