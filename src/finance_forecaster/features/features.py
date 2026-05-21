import numpy as np
import pandas as pd


def add_log_returns(df: pd.DataFrame) -> pd.DataFrame:
    """Add log return features."""
    df = df.copy()
    df["log_price"] = np.log(df["price"])
    df["log_return"] = df["log_price"].diff()
    df["log_return_squared"] = df["log_return"] ** 2
    return df


def add_lstm_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add technical and volatility features for LSTM model."""
    df = df.copy()

    # Lag features
    for lag in [1, 2, 3, 5]:
        df[f"return_lag_{lag}"] = df["log_return"].shift(lag)

    # Momentum and rolling stats
    df["momentum_3"] = df["log_return"].rolling(3).sum()
    df["momentum_5"] = df["log_return"].rolling(5).sum()
    df["momentum_10"] = df["log_return"].rolling(10).sum()

    df["rolling_mean_5"] = df["log_return"].rolling(5).mean()
    df["rolling_std_5"] = df["log_return"].rolling(5).std()
    df["rolling_mean_20"] = df["log_return"].rolling(20).mean()
    df["rolling_std_20"] = df["log_return"].rolling(20).std()
    df["volatility_ratio"] = df["rolling_std_5"] / df["rolling_std_20"]

    # Market regime related
    df["volume_momentum_5"] = df["volume"].rolling(5).mean() if "volume" in df.columns else 0

    df["target"] = (df["log_return"].shift(-1) > 0).astype(int)

    return df
