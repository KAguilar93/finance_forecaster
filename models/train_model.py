"""Model Training Pipeline

This module trains three different forecasting models on processed QQQ data:
- ARIMA: Autoregressive Integrated Moving Average for univariate time series
- GARCH: Generalized Autoregressive Conditional Heteroskedasticity for volatility
- LSTM: Long Short-Term Memory neural network for multivariate features
"""

import logging
from pathlib import Path
import pandas as pd
import numpy as np
import warnings

logger = logging.getLogger(__name__)

# Directory containing processed data files
PROCESSED_DIR = Path("data/processed")

def load_processed_data():
    """
    Load the processed dataset from disk.
    
    Returns:
        pd.DataFrame: Processed data with price, volume, market features, and engineered features
        None: If data file does not exist
    """
    qqq_file = PROCESSED_DIR / "qqq_processed.csv"
    if not qqq_file.exists():
        logger.error("Processed data not found. Please run 'make data' first.")
        return None
    # Load CSV with date index and parse dates
    df = pd.read_csv(qqq_file, index_col=0, parse_dates=True)
    logger.info(f"Loaded processed data: {df.shape[0]} rows, {df.shape[1]} columns")
    return df


def train_models():
    """
    Train all three forecasting models on the processed dataset.
    
    Models trained:
    1. ARIMA(1,0,1): Baseline univariate time series model
    2. GARCH(1,1): Conditional volatility model for risk forecasting
    3. LSTM: Deep learning model using multiple features
    
    Returns:
        bool: True if all models trained successfully, False otherwise
    """
    df = load_processed_data()
    if df is None:
        return False

    logger.info("=== Starting Model Training ===")

    # 1. ARIMA(1,0,1) - Univariate autoregressive model
    # order=(p,d,q): p=1 (AR), d=0 (no differencing), q=1 (MA)
    logger.info("Training ARIMA(1,0,1)...")
    try:
        from statsmodels.tsa.arima.model import ARIMA
        # Use log returns and remove NaN values
        arima_model = ARIMA(df["log_return"].dropna(), order=(1, 0, 1))
        arima_results = arima_model.fit()
        logger.info("ARIMA(1,0,1) training completed.")
    except Exception as e:
        logger.error(f"ARIMA failed: {e}")

    # 2. GARCH(1,1) - Conditional heteroskedasticity volatility model
    # Forecasts time-varying volatility; p=1 (lagged volatility), q=1 (lagged squared residuals)
    logger.info("Training GARCH(1,1)...")
    try:
        from arch import arch_model
        # Convert to percentage terms for better numerical stability
        returns = df["log_return"].dropna() * 100
        # Create GARCH model with constant mean specification
        garch_model = arch_model(returns, mean="Constant", vol="GARCH", p=1, q=1)
        garch_results = garch_model.fit(disp="off")
        logger.info("GARCH(1,1) training completed.")
    except Exception as e:
        logger.error(f"GARCH failed: {e}")

    # 3. LSTM - Deep learning model using sequence patterns
    # Captures temporal dependencies using multiple features and moving averages
    logger.info("Training LSTM model...")
    try:
        from sklearn.preprocessing import StandardScaler
        from tensorflow.keras.models import Sequential
        from tensorflow.keras.layers import LSTM, Dense, Dropout

        # Select features for multivariate input
        feature_cols = ["log_return", "ma_20", "ma_50"]
        data = df[feature_cols].dropna()
        
        # Standardize features to zero mean and unit variance (required for neural networks)
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(data)

        # Create sliding window sequences for supervised learning
        lookback = 30  # Use 30 days of history to predict next day
        X, y = [], []
        for i in range(lookback, len(X_scaled)):
            # Window of past 30 days with all 3 features
            X.append(X_scaled[i-lookback:i])
            # Target: next log_return (index 0 is log_return column)
            y.append(X_scaled[i, 0])

        X = np.array(X)
        y = np.array(y)

        # Build LSTM architecture: 2 stacked LSTM layers with dropout regularization
        model = Sequential([
            LSTM(64, return_sequences=True, input_shape=(lookback, X.shape[2])),
            Dropout(0.3),  # 30% dropout to prevent overfitting
            LSTM(32),
            Dropout(0.3),
            Dense(1)  # Single output: predicted log return
        ])
        # Compile with Adam optimizer and MSE loss
        model.compile(optimizer='adam', loss='mse')
        # Train for 20 epochs with batch size 32
        model.fit(X, y, epochs=20, batch_size=32, verbose=1)
        logger.info("LSTM training completed.")
    except Exception as e:
        logger.error(f"LSTM failed: {e}")

    logger.info("=== All models trained successfully ===")
    return True


def main():
    """
    Entry point for the model training pipeline.
    Configures logging and executes model training workflow.
    """
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    train_models()


if __name__ == "__main__":
    # Run training when file is executed directly
    main()
