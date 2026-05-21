"""Model training entrypoint."""

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

DATA_PATH = Path("data/raw/qqq_raw.csv")


def load_data(data_path: Path) -> pd.DataFrame:
    """Load and prepare price data from a yfinance CSV."""
    df = pd.read_csv(data_path, index_col=0, parse_dates=True, skiprows=[1, 2])
    if "Close" in df.columns and "price" not in df.columns:
        df = df.rename(columns={"Close": "price"})
    return add_log_returns(df).dropna()


def evaluate_arima(
    df: pd.DataFrame,
    order: tuple[int, int, int],
    test_split: float,
) -> tuple[object, dict[str, float], dict[str, float]]:
    """Fit ARIMA on train split, evaluate rolling 1-step forecasts on test split."""
    from statsmodels.tsa.arima.model import ARIMA  # noqa: PLC0415

    returns_pct = df["log_return"].dropna() * 100
    split = int(len(returns_pct) * (1 - test_split))
    train_series = returns_pct.iloc[:split]
    test_dates = returns_pct.index[split:]

    logger.info("Train: %d days | Test: %d days", split, len(test_dates))

    warnings.filterwarnings("ignore", module="statsmodels")
    model = ARIMA(train_series, order=order)
    fitted = model.fit()

    forecasts, actuals = [], []
    full_series = train_series.copy()
    for date in test_dates:
        m = ARIMA(full_series, order=order)
        r = m.fit()
        forecasts.append(float(r.forecast(steps=1).iloc[0]))
        actual = float(returns_pct.loc[date])
        actuals.append(actual)
        full_series = pd.concat([full_series, pd.Series([actual], index=[date])])

    forecasts_arr = np.array(forecasts)
    actuals_arr = np.array(actuals)

    clf = classification_report((actuals_arr > 0).astype(int), (forecasts_arr > 0).astype(int))
    reg = regression_report(actuals_arr, forecasts_arr)

    return fitted, clf, reg


def prepare_lstm_sequences(
    df: pd.DataFrame,
    lookback: int,
    test_split: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, StandardScaler]:
    """Build lookback sequences for LSTM training."""
    df_feat = add_lstm_features(df).dropna()
    feature_cols = [c for c in df_feat.columns if c not in ("target", "price", "log_price")]
    features = df_feat[feature_cols].values
    target = df_feat["target"].values

    split = int(len(features) * (1 - test_split))
    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(features)

    X, y = [], []
    for i in range(lookback, len(features_scaled)):
        X.append(features_scaled[i - lookback : i])
        y.append(target[i])
    X_arr = np.array(X)  # type: ignore[assignment]
    y_arr = np.array(y)  # type: ignore[assignment]

    seq_split = split - lookback
    return X_arr[:seq_split], X_arr[seq_split:], y_arr[:seq_split], y_arr[seq_split:], scaler


def evaluate_lstm(
    df: pd.DataFrame,
    epochs: int,
    batch_size: int,
    lookback: int,
    test_split: float,
) -> tuple[Any, dict[str, float]]:
    """Train LSTM and return fitted model + classification metrics."""
    from finance_forecaster.models.lstm import train_lstm_model  # noqa: PLC0415

    X_train, X_test, y_train, y_test, _ = prepare_lstm_sequences(df, lookback, test_split)
    logger.info(
        "LSTM sequences | Train: %d | Test: %d | Shape: %s",
        len(X_train),
        len(X_test),
        X_train.shape[1:],
    )

    model, _, _, y_pred = train_lstm_model(
        X_train,
        y_train,
        X_test,
        y_test,
        epochs=epochs,
        batch_size=batch_size,
    )
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
    """Train the selected model and log the run to MLflow."""
    model_dir.mkdir(parents=True, exist_ok=True)

    mlflow_cfg = DEFAULT_CONFIG.mlflow
    mlflow.set_tracking_uri(mlflow_cfg.tracking_uri)
    mlflow.set_experiment(mlflow_cfg.experiment_name)

    if mlflow_cfg.system_metrics:
        mlflow.enable_system_metrics_logging()

    logger.info("Loading data from %s", data_path)
    df = load_data(data_path)
    logger.info("Loaded %d trading days", len(df))

    with mlflow.start_run():
        mlflow.log_param("model_type", model_type)
        mlflow.log_param("test_split", test_split)
        mlflow.log_param("n_observations", len(df))
        mlflow.log_param("ticker", "QQQ")

        if model_type == "arima":
            order = (arima_p, arima_d, arima_q)
            mlflow.log_params({"arima_p": arima_p, "arima_d": arima_d, "arima_q": arima_q})
            logger.info("Fitting ARIMA%s...", order)
            fitted, clf, reg = evaluate_arima(df, order, test_split)

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

            model_path = model_dir / f"arima_{arima_p}_{arima_d}_{arima_q}.joblib"
            joblib.dump(fitted, model_path)
            mlflow.log_artifact(str(model_path))
            logger.info("ARIMA%s | accuracy=%.1f%%", order, clf["accuracy"] * 100)

        elif model_type == "lstm":
            mlflow.log_params({"epochs": epochs, "batch_size": batch_size, "lookback": lookback})
            logger.info("Training LSTM (epochs=%d, batch=%d, lookback=%d)...", epochs, batch_size, lookback)
            model, clf = evaluate_lstm(df, epochs, batch_size, lookback, test_split)

            mlflow.log_metrics(
                {
                    "directional_accuracy": clf["accuracy"],
                    "precision": clf["precision"],
                    "recall": clf["recall"],
                    "f1": clf["f1"],
                }
            )

            model_path = model_dir / "lstm_model.keras"
            model.save(str(model_path))
            mlflow.log_artifact(str(model_path))
            logger.info("LSTM | accuracy=%.1f%%", clf["accuracy"] * 100)

        else:
            raise ValueError(f"Unknown model_type '{model_type}'. Choose 'arima' or 'lstm'.")

        logger.info("Model saved -> %s", model_path)
        logger.info("MLflow run logged to experiment '%s'", mlflow_cfg.experiment_name)


def main() -> None:
    """CLI entrypoint for model training."""
    parser = argparse.ArgumentParser(description="Train a model and log to MLflow")
    parser.add_argument("--data-path", type=Path, default=DATA_PATH)
    parser.add_argument("--model-dir", type=Path, default=MODELS_DIR)
    parser.add_argument("--model-type", choices=["arima", "lstm"], default="arima")
    parser.add_argument("--test-split", type=float, default=0.2)
    # ARIMA args
    parser.add_argument("--arima-p", type=int, default=1)
    parser.add_argument("--arima-d", type=int, default=0)
    parser.add_argument("--arima-q", type=int, default=1)
    # LSTM args
    parser.add_argument("--epochs", type=int, default=DEFAULT_CONFIG.training.epochs)
    parser.add_argument("--batch-size", type=int, default=DEFAULT_CONFIG.training.batch_size)
    parser.add_argument("--lookback", type=int, default=30)
    parser.add_argument("--seed", type=int, default=DEFAULT_CONFIG.training.seed)
    args = parser.parse_args()

    setup_logging()
    set_seed(args.seed)

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
