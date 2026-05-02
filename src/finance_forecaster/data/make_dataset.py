"""Download raw market data and create a processed modeling dataset.

Pipeline:
1. Fetch historical ticker data from yfinance
2. Save raw data to data/raw/
3. Create next-day direction target
4. Save processed data to data/processed/
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
import yfinance as yf

from finance_forecaster.config import PROCESSED_DATA_DIR, RAW_DATA_DIR
from finance_forecaster.logging_config import get_logger, setup_logging

logger = get_logger(__name__)


def fetch_raw_data(ticker: str, start_date: str, output_file: Path) -> Path:
    """Download historical market data from yfinance and save it as raw CSV."""
    logger.info("Downloading %s data from yfinance starting at %s", ticker, start_date)

    output_file.parent.mkdir(parents=True, exist_ok=True)

    df = yf.download(ticker, start=start_date, progress=False, auto_adjust=False)

    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    if df.empty:
        raise ValueError(f"No data returned from yfinance for ticker: {ticker}")

    df = df.reset_index()
    df.to_csv(output_file, index=False)

    logger.info("Saved raw data to %s", output_file)
    logger.info("Raw dataset shape: %s rows, %s columns", df.shape[0], df.shape[1])

    return output_file


def process_data(input_file: Path, output_file: Path) -> Path:
    """Create a processed dataset for next-day direction prediction."""
    logger.info("Reading raw data from %s", input_file)

    if not input_file.exists():
        raise FileNotFoundError(f"Raw input file does not exist: {input_file}")

    df = pd.read_csv(input_file)

    required_columns = {"Date", "Open", "High", "Low", "Close", "Volume"}
    missing_columns = required_columns.difference(df.columns)

    if missing_columns:
        raise ValueError(f"Missing required columns: {sorted(missing_columns)}")

    df = df.copy()
    df["Date"] = pd.to_datetime(df["Date"])
    df = df.sort_values("Date").reset_index(drop=True)

    df["daily_return"] = df["Close"].pct_change()
    df["next_close"] = df["Close"].shift(-1)
    df["target_next_day_direction"] = (df["next_close"] > df["Close"]).astype(int)

    df = df.dropna().reset_index(drop=True)

    output_file.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_file, index=False)

    logger.info("Saved processed data to %s", output_file)
    logger.info("Processed dataset shape: %s rows, %s columns", df.shape[0], df.shape[1])

    return output_file


def main() -> None:
    """CLI entrypoint for the full data pipeline."""
    parser = argparse.ArgumentParser(
        description="Fetch raw market data and create processed model dataset"
    )
    parser.add_argument("--ticker", type=str, default="QQQ")
    parser.add_argument("--start-date", type=str, default="2015-01-01")
    parser.add_argument(
        "--raw-output",
        type=Path,
        default=RAW_DATA_DIR / "qqq.csv",
    )
    parser.add_argument(
        "--processed-output",
        type=Path,
        default=PROCESSED_DATA_DIR / "qqq_processed.csv",
    )
    args = parser.parse_args()

    setup_logging()

    raw_file = fetch_raw_data(
        ticker=args.ticker,
        start_date=args.start_date,
        output_file=args.raw_output,
    )

    process_data(
        input_file=raw_file,
        output_file=args.processed_output,
    )

    logger.info("Data pipeline complete")


if __name__ == "__main__":
    main()
