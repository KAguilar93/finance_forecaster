"""Model inference entrypoint."""

from __future__ import annotations

import argparse
import warnings
from pathlib import Path

import pandas as pd

from finance_forecaster.features.features import add_log_returns
from finance_forecaster.logging_config import get_logger, setup_logging
from finance_forecaster.models.arima import generate_rolling_arima_forecasts

logger = get_logger(__name__)

# Number of recent trading days to use for the rolling ARIMA forecast window
FORECAST_WINDOW = 30


def predict(input_path: Path, output_path: Path) -> None:
    """Generate next-day directional predictions from input price data.

    Loads a CSV with a 'price' column (or Close column from yfinance),
    computes log returns, runs a rolling ARIMA(1,0,1) forecast over the
    last FORECAST_WINDOW days, and writes predictions to output_path.
    """
    logger.info("Loading input data from %s", input_path)
    # yfinance CSVs have a 3-row multi-level header (Price/Ticker/Date);
    # skiprows=[1,2] drops the Ticker and Date label rows.
    df = pd.read_csv(input_path, index_col=0, parse_dates=True, skiprows=[1, 2])

    # Normalise column name — yfinance raw export uses 'price' or 'Close'
    if "Close" in df.columns and "price" not in df.columns:
        df = df.rename(columns={"Close": "price"})
    if "price" not in df.columns:
        raise ValueError(f"Input file must have a 'price' or 'Close' column. Got: {list(df.columns)}")

    df = add_log_returns(df).dropna()
    logger.info("Loaded %d trading days", len(df))

    # Roll ARIMA over the last FORECAST_WINDOW days to produce forecasts
    eval_dates = df.index[-FORECAST_WINDOW:]
    logger.info("Running ARIMA(1,0,1) rolling forecast over %d days...", FORECAST_WINDOW)
    warnings.filterwarnings("ignore", module="statsmodels")
    forecasts = generate_rolling_arima_forecasts(df, eval_dates)
    forecasts = forecasts.set_index("date")

    # Align forecasts with actual returns for evaluation.
    # ARIMA trains on data through day d and forecasts d+1, so we compare
    # the forecast at d against the actual return at d+1 (shift(-1)).
    results = df.loc[eval_dates].copy()
    results["arima_forecast_pct"] = forecasts["arima_mean_forecast_percent"]
    results["predicted_direction"] = (results["arima_forecast_pct"] > 0).map({True: "UP", False: "DOWN"})
    results["actual_direction"] = (results["log_return"].shift(-1) > 0).map({True: "UP", False: "DOWN"})
    results = results.dropna(subset=["actual_direction"])
    results["correct"] = results["predicted_direction"] == results["actual_direction"]

    # One-step-ahead forecast for the next trading day (beyond the dataset)
    logger.info("Generating next-day forecast...")
    from statsmodels.tsa.arima.model import ARIMA as _ARIMA  # noqa: PLC0415

    returns_pct = df["log_return"].dropna() * 100
    model = _ARIMA(returns_pct, order=(1, 0, 1))
    result = model.fit()
    next_day_forecast = float(result.forecast(steps=1).iloc[0])
    next_day_direction = "UP" if next_day_forecast > 0 else "DOWN"

    logger.info(
        "Next-day forecast: %.4f%% -> %s",
        next_day_forecast,
        next_day_direction,
    )

    # Write output
    output_path.parent.mkdir(parents=True, exist_ok=True)
    results.to_csv(output_path)
    logger.info("Predictions written to %s", output_path)

    # Print summary
    accuracy = results["correct"].mean()
    logger.info(
        "Rolling window accuracy (%d days): %.1f%%",
        FORECAST_WINDOW,
        accuracy * 100,
    )
    logger.info("Next trading day prediction: %s (%.4f%%)", next_day_direction, next_day_forecast)


def main() -> None:
    """CLI entrypoint for batch prediction."""
    parser = argparse.ArgumentParser(description="Generate next-day directional predictions")
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("data/raw/qqq_raw.csv"),
        help="Path to CSV with price data",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("predictions") / "predictions.csv",
        help="Path to write predictions CSV",
    )
    args = parser.parse_args()

    setup_logging()
    predict(args.input, args.output)
    logger.info("Prediction complete")


if __name__ == "__main__":
    main()
