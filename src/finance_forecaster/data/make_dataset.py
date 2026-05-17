import logging
from pathlib import Path
import pandas as pd
import numpy as np
import yfinance as yf

logger = logging.getLogger(__name__)

RAW_DIR = Path("data/raw")
PROCESSED_DIR = Path("data/processed")

def download_stock_data(ticker="QQQ", start="2015-01-01", end="2026-05-15"):
    logger.info(f"Downloading main ticker: {ticker}")
    data = yf.download(ticker, start=start, end=end, auto_adjust=True, progress=False)
    
    if data.empty:
        logger.error("No data returned from yfinance")
        return pd.DataFrame()
    
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
            logger.warning(f"No data found for {ticker}")
            continue
        
        if isinstance(data.columns, pd.MultiIndex):
            close = data["Close"]
            if isinstance(close, pd.DataFrame):
                close = close.iloc[:, 0]
        else:
            close = data["Close"]
        
        feature_data[f"{name}_close"] = close
        feature_data[f"{name}_return"] = np.log(close).diff()
    
    feature_data.dropna(inplace=True)
    
    # Feature engineering after log returns
    feature_data["iwm_spy_relative_strength"] = feature_data["iwm_close"] / feature_data["spy_close"]
    feature_data["qqq_spy_market_strength"] = feature_data["xlk_close"] / feature_data["spy_close"]
    feature_data["tech_vs_staples"] = feature_data["xlk_close"] / feature_data["xlp_close"]
    feature_data["discretionary_vs_staples"] = feature_data["xly_close"] / feature_data["xlp_close"]
    feature_data["financials_vs_market"] = feature_data["xlf_close"] / feature_data["spy_close"]
    feature_data["industrials_vs_market"] = feature_data["xli_close"] / feature_data["spy_close"]
    feature_data["credit_risk_appetite"] = feature_data["hyg_close"] / feature_data["tlt_close"]
    feature_data["gold_vs_market"] = feature_data["gld_close"] / feature_data["spy_close"]
    
    logger.info(f"Created {feature_data.shape[1]} market feature columns")
    return feature_data


def process_data():
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    # Raw backups
    qqq_raw = RAW_DIR / "qqq_raw.csv"
    market_raw = RAW_DIR / "market_features.csv"

    # Processed files
    qqq_processed = PROCESSED_DIR / "qqq_processed.csv"
    market_processed = PROCESSED_DIR / "market_features_processed.csv"

    # Download raw backups if missing
    if not qqq_raw.exists():
        df = download_stock_data()
        df.to_csv(qqq_raw)
        logger.info(f"Saved raw backup: {qqq_raw.name}")
    else:
        logger.info("Raw QQQ backup exists.")

    if not market_raw.exists():
        market_df = download_external_market_features()
        market_df.to_csv(market_raw)
        logger.info(f"Saved raw backup: {market_raw.name}")
    else:
        logger.info("Raw market features backup exists.")

    # Save market features to processed folder with new name
    if not market_processed.exists():
        market_df = pd.read_csv(market_raw, index_col=0, parse_dates=True)
        market_df.to_csv(market_processed)
        logger.info(f"Saved market features to processed: {market_processed.name}")
    else:
        logger.info("Market features already in processed folder.")

    # Create main processed file
    if not qqq_processed.exists():
        logger.info("Creating merged processed dataset...")
        qqq = pd.read_csv(qqq_raw, index_col=0, parse_dates=True)
        market = pd.read_csv(market_raw, index_col=0, parse_dates=True)

        df = qqq.join(market, how="left")
        df.ffill(inplace=True)

        df["log_return"] = np.log(df["price"]).diff()
        df.dropna(inplace=True)

        df.to_csv(qqq_processed)
        logger.info(f"Processed data saved → {qqq_processed.name} ({df.shape[0]} rows)")
    else:
        logger.info("Processed data already exists.")

    logger.info("Data pipeline complete")
    return True


def main():
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    process_data()


if __name__ == "__main__":
    main()
