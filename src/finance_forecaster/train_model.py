"""Model training entrypoint.

This module provides training pipelines for two time series forecasting models:
- ARIMA: Autoregressive Integrated Moving Average for statistical time series modeling
- LSTM: Long Short-Term Memory neural network for deep learning-based forecasting

Both models predict directional movement (up/down) of QQQ returns on the next trading day.
Training runs are logged to MLflow for experiment tracking and model versioning.
"""

from __future__ import annotations

import argparse
import warnings
from pathlib import Path
from typing import Any

import joblib
import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from finance_forecaster.config import DEFAULT_CONFIG, MODELS_DIR
from finance_forecaster.evaluation.metrics import classification_report, regression_report
from finance_forecaster.features.features import add_log_returns, add_lstm_features
from finance_forecaster.logging_config import get_logger, setup_logging
from finance_forecaster.utils.seed import set_seed

logger = get_logger(__name__)

# Path to raw QQQ price data from Yahoo Finance
DATA_PATH = Path("data/raw/qqq_raw.csv")


def load_data(data_path: Path) -> pd.DataFrame:
    """Load and prepare price data from a yfinance CSV.
    
    Args:
        data_path: Path to CSV file with OHLCV data from yfinance
    
    Returns:
        DataFrame with columns: price, log_return indexed by date
    """
    # Read CSV, using date column as index and skipping metadata rows
    df = pd.read_csv(data_path, index_col=0, parse_dates=True, skiprows=[1, 2])
    
    # Rename 'Close' to 'price' for consistency across data sources
    if "Close" in df.columns and "price" not in df.columns:
        df = df.rename(columns={"Close": "price"})
    
    # Add log returns feature and remove any NaN values
    return add_log_returns(df).dropna()


def evaluate_arima(
    df: pd.DataFrame,
    order: tuple[int, int, int],
    test_split: float,
) -> tuple[object, dict[str, float], dict[str, float]]:
    """Fit ARIMA on train split, evaluate rolling 1-step forecasts on test split.
    
    Uses a walk-forward validation approach where the model is retrained for each
    test observation, simulating real-world deployment where new data arrives daily.
    
    Args:
        df: DataFrame with columns ['log_return', 'price']
        order: ARIMA (p, d, q) parameters
        test_split: Fraction of data to use for testing (e.g., 0.2 = 80/20 split)
    
    Returns:
        Tuple of (fitted_model, classification_metrics, regression_metrics)
    """
    from statsmodels.tsa.arima.model import ARIMA  # noqa: PLC0415

    # Convert log returns to percentage for interpretability
    returns_pct = df["log_return"].dropna() * 100
    # Calculate train/test split index
    split = int(len(returns_pct) * (1 - test_split))
    train_series = returns_pct.iloc[:split]
    test_dates = returns_pct.index[split:]

    logger.info("Train: %d days | Test: %d days", split, len(test_dates))

    # Suppress statsmodels convergence warnings
    warnings.filterwarnings("ignore", module="statsmodels")
    # Fit initial model on full training set
    model = ARIMA(train_series, order=order)
    fitted = model.fit()

    # Walk-forward validation: retrain for each test point with expanding window
    forecasts, actuals = [], []
    full_series = train_series.copy()
    for date in test_dates:
        # Retrain model with data up to previous day
        m = ARIMA(full_series, order=order)
        r = m.fit()
        # Make 1-step ahead forecast
        forecasts.append(float(r.forecast(steps=1).iloc[0]))
        # Record actual return on this date
        actual = float(returns_pct.loc[date])
        actuals.append(actual)
        # Add actual to series for next iteration (expanding window)
        full_series = pd.concat([full_series, pd.Series([actual], index=[date])])

    # Convert to arrays for metric calculation
    forecasts_arr = np.array(forecasts)
    actuals_arr = np.array(actuals)

    # Classification metrics based on direction (positive/negative return)
    clf = classification_report((actuals_arr > 0).astype(int), (forecasts_arr > 0).astype(int))
    # Regression metrics on actual return values
    reg = regression_report(actuals_arr, forecasts_arr)

    return fitted, clf, reg


def prepare_lstm_sequences(
    df: pd.DataFrame,
    lookback: int,
    test_split: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, StandardScaler]:
    """Build lookback sequences for LSTM training.
    
    Transforms 1D time series into 2D sequences suitable for LSTM:
    Input shape: (n_samples, lookback, n_features)
    Each sample contains 'lookback' historical observations.
    
    Args:
        df: DataFrame with OHLCV and engineered features
        lookback: Number of historical days to include in each sequence
        test_split: Fraction of data for testing
    
    Returns:
        Tuple of (X_train, X_test, y_train, y_test, scaler) where X arrays
        have shape (n_samples, lookback, n_features)
    """
    # Add LSTM-specific features and remove NaN values
    df_feat = add_lstm_features(df).dropna()
    # Extract feature columns (exclude target and price columns)
    feature_cols = [c for c in df_feat.columns if c not in ("target", "price", "log_price")]
    features = df_feat[feature_cols].values
    target = df_feat["target"].values

    # Calculate train/test split point
    split = int(len(features) * (1 - test_split))
    # Fit scaler on training data only to prevent data leakage
    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(features)

    # Create sliding windows of lookback length
    X, y = [], []
    for i in range(lookback, len(features_scaled)):
        # Each X sample is a window of lookback consecutive observations
        X.append(features_scaled[i - lookback : i])
        # Target is the label for the day after this window
        y.append(target[i])
    X_arr = np.array(X)  # type: ignore[assignment]
    y_arr = np.array(y)  # type: ignore[assignment]

    # Adjust split point to account for lookback sequences that consume training data
    seq_split = split - lookback
    # Return train/test splits along with fitted scaler for inference
    return X_arr[:seq_split], X_arr[seq_split:], y_arr[:seq_split], y_arr[seq_split:], scaler


def evaluate_lstm(
    df: pd.DataFrame,
    epochs: int,
    batch_size: int,
    lookback: int,
    test_split: float,
) -> tuple[Any, dict[str, float]]:
    """Train LSTM and return fitted model + classification metrics.
    
    Args:
        df: DataFrame with OHLCV and engineered features
        epochs: Number of training epochs
        batch_size: Number of samples per gradient update
        lookback: Length of input sequences (historical days)
        test_split: Fraction of data for testing
    
    Returns:
        Tuple of (trained_model, classification_metrics_dict)
    """
    from finance_forecaster.models.lstm import train_lstm_model  # noqa: PLC0415

    # Prepare sequences and split data
    X_train, X_test, y_train, y_test, _ = prepare_lstm_sequences(df, lookback, test_split)
    # Log sequence statistics
    logger.info(
        "LSTM sequences | Train: %d | Test: %d | Shape: %s",
        len(X_train),
        len(X_test),
        X_train.shape[1:],
    )

    # Train model and get predictions
    model, _, _, y_pred = train_lstm_model(
        X_train,
        y_train,
        X_test,
        y_test,
        epochs=epochs,
        batch_size=batch_size,
    )
    # Calculate directional classification metrics
    clf = classification_report(y_test.astype(int), y_pred.astype(int))
    return model, clf


def train(
    data_path: Path,
    model_dir: Path,
    model_type: str,
    arima_p: int,
    arima_d: int,
    arima_q: int,
    epochs: int,
    batch_size: int,
    lookback: int,
    test_split: float,
) -> None:
    """Train the selected model and log the run to MLflow.
    
    Supports two model types:
    - 'arima': Classical statistical time series model
    - 'lstm': Deep learning recurrent neural network
    
    All metrics and artifacts are logged to MLflow for experiment tracking.
    
    Args:
        data_path: Path to input CSV file
        model_dir: Directory where trained model will be saved
        model_type: Either 'arima' or 'lstm'
        arima_p, arima_d, arima_q: ARIMA(p,d,q) parameters
        epochs: Number of training epochs (LSTM only)
        batch_size: Training batch size (LSTM only)
        lookback: Sequence length in days (LSTM only)
        test_split: Fraction of data for testing (e.g., 0.2)
    """
    # Ensure model directory exists
    model_dir.mkdir(parents=True, exist_ok=True)

    # Configure MLflow tracking
    mlflow_cfg = DEFAULT_CONFIG.mlflow
    mlflow.set_tracking_uri(mlflow_cfg.tracking_uri)
    mlflow.set_experiment(mlflow_cfg.experiment_name)

    # Enable system metrics collection (CPU, memory, etc.)
    if mlflow_cfg.system_metrics:
        mlflow.enable_system_metrics_logging()

    # Load and prepare data
    logger.info("Loading data from %s", data_path)
    df = load_data(data_path)
    logger.info("Loaded %d trading days", len(df))

    # Start MLflow run to track this training session
    with mlflow.start_run():
        # Log global parameters
        mlflow.log_param("model_type", model_type)
        mlflow.log_param("test_split", test_split)
        mlflow.log_param("n_observations", len(df))
        mlflow.log_param("ticker", "QQQ")

        # Train ARIMA model if selected
        if model_type == "arima":
            # Set ARIMA parameters
            order = (arima_p, arima_d, arima_q)
            mlflow.log_params({"arima_p": arima_p, "arima_d": arima_d, "arima_q": arima_q})
            logger.info("Fitting ARIMA%s...", order)
            
            # Fit and evaluate ARIMA model
            fitted, clf, reg = evaluate_arima(df, order, test_split)

            # Log all metrics (both classification and regression)
            mlflow.log_metrics(
                {
                    "directional_accuracy": clf["accuracy"],
                    "precision": clf["precision"],
                    "recall": clf["recall"],
                    "f1": clf["f1"],
                    "mae": reg["mae"],
                    "rmse": reg["rmse"],
                    "r2": reg["r2"],
                }
            )

            # Save model to disk and log as artifact
            model_path = model_dir / f"arima_{arima_p}_{arima_d}_{arima_q}.joblib"
            joblib.dump(fitted, model_path)
            mlflow.log_artifact(str(model_path))
            logger.info("ARIMA%s | accuracy=%.1f%%", order, clf["accuracy"] * 100)

        elif model_type == "lstm":
            # Log LSTM hyperparameters
            mlflow.log_params({"epochs": epochs, "batch_size": batch_size, "lookback": lookback})
            logger.info("Training LSTM (epochs=%d, batch=%d, lookback=%d)...", epochs, batch_size, lookback)
            
            # Train and evaluate LSTM model
            model, clf = evaluate_lstm(df, epochs, batch_size, lookback, test_split)

            # Log classification metrics
            mlflow.log_metrics(
                {
                    "directional_accuracy": clf["accuracy"],
                    "precision": clf["precision"],
                    "recall": clf["recall"],
                    "f1": clf["f1"],
                }
            )

            # Save Keras model and log as artifact
            model_path = model_dir / "lstm_model.keras"
            model.save(str(model_path))
            mlflow.log_artifact(str(model_path))
            logger.info("LSTM | accuracy=%.1f%%", clf["accuracy"] * 100)

        else:
            # Handle invalid model selection
            raise ValueError(f"Unknown model_type '{model_type}'. Choose 'arima' or 'lstm'.")

        # Log completion status
        logger.info("Model saved -> %s", model_path)
        logger.info("MLflow run logged to experiment '%s'", mlflow_cfg.experiment_name)


def main() -> None:
    """CLI entrypoint for model training.
    
    Supports training both ARIMA and LSTM models with customizable parameters.
    Run with --help to see all available options.
    
    Example usage:
        python train_model.py --model-type arima --arima-p 1 --arima-d 1 --arima-q 1
        python train_model.py --model-type lstm --epochs 100 --batch-size 32 --lookback 30
    """
    # Configure argument parser with all training parameters
    parser = argparse.ArgumentParser(description="Train a model and log to MLflow")
    parser.add_argument("--data-path", type=Path, default=DATA_PATH)
    parser.add_argument("--model-dir", type=Path, default=MODELS_DIR)
    parser.add_argument("--model-type", choices=["arima", "lstm"], default="arima")
    parser.add_argument("--test-split", type=float, default=0.2, help="Fraction of data for testing")
    
    # ARIMA-specific arguments
    parser.add_argument("--arima-p", type=int, default=1, help="AR (autoregressive) order")
    parser.add_argument("--arima-d", type=int, default=0, help="I (differencing) order")
    parser.add_argument("--arima-q", type=int, default=1, help="MA (moving average) order")
    
    # LSTM-specific arguments
    parser.add_argument("--epochs", type=int, default=DEFAULT_CONFIG.training.epochs)
    parser.add_argument("--batch-size", type=int, default=DEFAULT_CONFIG.training.batch_size)
    parser.add_argument("--lookback", type=int, default=30, help="Historical days in each sequence")
    parser.add_argument("--seed", type=int, default=DEFAULT_CONFIG.training.seed, help="Random seed for reproducibility")
    args = parser.parse_args()

    # Initialize logging and random seed for reproducibility
    setup_logging()
    set_seed(args.seed)

    # Execute training with parsed arguments
    train(
        args.data_path,
        args.model_dir,
        args.model_type,
        args.arima_p,
        args.arima_d,
        args.arima_q,
        args.epochs,
        args.batch_size,
        args.lookback,
        args.test_split,
    )
    logger.info("Training complete")


if __name__ == "__main__":
    main()
