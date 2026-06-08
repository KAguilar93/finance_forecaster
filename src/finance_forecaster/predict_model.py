"""Model inference entrypoint.

This module provides prediction functionality for next-day directional movement of QQQ.
It loads historical price data, computes a rolling ARIMA(1,0,1) forecast over recent data,
evaluates accuracy on a historical window, and generates a next-day prediction.

The workflow:
1. Load and normalize price data from CSV (from yfinance)
2. Calculate log returns
3. Run rolling ARIMA forecasts over FORECAST_WINDOW recent days
4. Compare predictions against actual returns to compute accuracy
5. Train final ARIMA model on all historical data for next-day forecast
"""

from __future__ import annotations

import argparse
import warnings
from pathlib import Path

import pandas as pd

from finance_forecaster.features.features import add_log_returns
from finance_forecaster.logging_config import get_logger, setup_logging
from finance_forecaster.models.arima import generate_rolling_arima_forecasts

logger = get_logger(__name__)

# Number of recent trading days used for rolling prediction evaluation
# Larger windows provide more evaluation data but use older/less relevant training sets
FORECAST_WINDOW = 30


def predict(input_path: Path, output_path: Path) -> None:
    """Generate next-day directional predictions from input price data.
    
    Pipeline:
    1. Load CSV with price/Close column, normalizing column names
    2. Calculate log returns for stationarity
    3. Run rolling ARIMA(1,0,1) forecasts over FORECAST_WINDOW recent days
    4. Evaluate in-sample accuracy by comparing forecasts to actual returns
    5. Train final model on all historical data for next-day prediction
    6. Write results to CSV with predictions and accuracy metrics
    
    Args:
        input_path: Path to CSV file containing price data (from yfinance format)
        output_path: Path where prediction results will be written
    """
    # Load data with yfinance multi-level header handling
    logger.info("Loading input data from %s", input_path)
    # yfinance CSVs have a 3-row multi-level header (Price/Ticker/Date);
    # skiprows=[1,2] drops the Ticker and Date label rows.
    df = pd.read_csv(input_path, index_col=0, parse_dates=True, skiprows=[1, 2])

    # Normalize column names for consistency (yfinance uses 'Close', we use 'price')
    if "Close" in df.columns and "price" not in df.columns:
        df = df.rename(columns={"Close": "price"})
    if "price" not in df.columns:
        raise ValueError(f"Input file must have a 'price' or 'Close' column. Got: {list(df.columns)}")

    # Add log returns (stationary feature for ARIMA) and remove NaN values
    df = add_log_returns(df).dropna()
    logger.info("Loaded %d trading days", len(df))

    # Setup rolling window validation: use last FORECAST_WINDOW days for evaluation
    eval_dates = df.index[-FORECAST_WINDOW:]
    logger.info("Running ARIMA(1,0,1) rolling forecast over %d days...", FORECAST_WINDOW)
    
    # Suppress statsmodels convergence warnings (expected for time series fitting)
    warnings.filterwarnings("ignore", module="statsmodels")
    # Generate rolling forecasts (model retrains for each date, using prior data only)
    forecasts = generate_rolling_arima_forecasts(df, eval_dates)
    forecasts = forecasts.set_index("date")

    # Align rolling forecasts with actual returns for accuracy evaluation
    # Note: ARIMA(d) trains on data through day d and forecasts day d+1,
    # so we align forecast at d with actual return on d+1 (using shift(-1))
    results = df.loc[eval_dates].copy()
    results["arima_forecast_pct"] = forecasts["arima_mean_forecast_percent"]
    
    # Convert forecast values to directional predictions (UP/DOWN)
    results["predicted_direction"] = (results["arima_forecast_pct"] > 0).map({True: "UP", False: "DOWN"})
    # Actual direction on next day (shift backward one row to align with forecast date)
    results["actual_direction"] = (results["log_return"].shift(-1) > 0).map({True: "UP", False: "DOWN"})
    
    # Remove last row which has no next-day actual value
    results = results.dropna(subset=["actual_direction"])
    # Check if prediction matches actual direction
    results["correct"] = results["predicted_direction"] == results["actual_direction"]

    # Generate next-day prediction by training on entire historical dataset
    logger.info("Generating next-day forecast...")
    from statsmodels.tsa.arima.model import ARIMA as _ARIMA  # noqa: PLC0415

    # Use all available data for maximum information in final forecast
    returns_pct = df["log_return"].dropna() * 100
    # Train ARIMA(1,0,1) on full dataset
    model = _ARIMA(returns_pct, order=(1, 0, 1))
    result = model.fit()
    # Make 1-step ahead forecast (next trading day)
    next_day_forecast = float(result.forecast(steps=1).iloc[0])
    # Classify as directional movement
    next_day_direction = "UP" if next_day_forecast > 0 else "DOWN"

    logger.info(
        "Next-day forecast: %.4f%% -> %s",
        next_day_forecast,
        next_day_direction,
    )

    # Save results and next-day prediction to CSV
    output_path.parent.mkdir(parents=True, exist_ok=True)
    results.to_csv(output_path)
    logger.info("Predictions written to %s", output_path)

    # Log accuracy metrics for evaluation
    accuracy = results["correct"].mean()
    logger.info(
        "Rolling window accuracy (%d days): %.1f%%",
        FORECAST_WINDOW,
        accuracy * 100,
    )
    # Report final prediction
    logger.info("Next trading day prediction: %s (%.4f%%)", next_day_direction, next_day_forecast)


def main() -> None:
    """CLI entrypoint for batch prediction.
    
    Generates next-day directional predictions for QQQ and evaluates accuracy
    on a rolling window of recent data.
    
    Example usage:
        python predict_model.py --input data/raw/qqq_raw.csv --output predictions/pred.csv
    """
    # Configure command-line arguments
    parser = argparse.ArgumentParser(description="Generate next-day directional predictions")
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("data/raw/qqq_raw.csv"),
        help="Path to CSV with price data (yfinance format)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("predictions") / "predictions.csv",
        help="Path to write predictions CSV with historical accuracy and next-day forecast",
    )
    args = parser.parse_args()

    # Setup logging and run prediction pipeline
    setup_logging()
    predict(args.input, args.output)
    logger.info("Prediction complete")


# Execute prediction pipeline when run as script
if __name__ == "__main__":
    main()

