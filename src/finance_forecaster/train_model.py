"""Model training entrypoint."""

from __future__ import annotations

import argparse
import warnings
from pathlib import Path

import joblib
import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd

from finance_forecaster.config import DEFAULT_CONFIG, MODELS_DIR
from finance_forecaster.evaluation.metrics import classification_report, regression_report
from finance_forecaster.features.features import add_log_returns
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

    # Final model fit on training data
    warnings.filterwarnings("ignore", module="statsmodels")
    model = ARIMA(train_series, order=order)
    fitted = model.fit()

    # Rolling 1-step-ahead evaluation on test set
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

    actual_dir = (actuals_arr > 0).astype(int)
    forecast_dir = (forecasts_arr > 0).astype(int)

    clf = classification_report(actual_dir, forecast_dir)
    reg = regression_report(actuals_arr, forecasts_arr)

    return fitted, clf, reg


def train(
    data_path: Path,
    model_dir: Path,
    arima_p: int,
    arima_d: int,
    arima_q: int,
    test_split: float,
) -> None:
    """Train an ARIMA model and log the run to MLflow."""
    model_dir.mkdir(parents=True, exist_ok=True)

    mlflow_cfg = DEFAULT_CONFIG.mlflow
    mlflow.set_tracking_uri(mlflow_cfg.tracking_uri)
    mlflow.set_experiment(mlflow_cfg.experiment_name)

    if mlflow_cfg.system_metrics:
        mlflow.enable_system_metrics_logging()

    order = (arima_p, arima_d, arima_q)
    logger.info("Loading data from %s", data_path)
    df = load_data(data_path)
    logger.info("Loaded %d trading days", len(df))

    with mlflow.start_run():
        mlflow.log_params(
            {
                "ticker": "QQQ",
                "arima_p": arima_p,
                "arima_d": arima_d,
                "arima_q": arima_q,
                "arima_order": str(order),
                "test_split": test_split,
                "n_observations": len(df),
            }
        )

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

        # Save and log model artifact
        model_path = model_dir / f"arima_{arima_p}_{arima_d}_{arima_q}.joblib"
        joblib.dump(fitted, model_path)
        mlflow.log_artifact(str(model_path))

        logger.info(
            "ARIMA%s | accuracy=%.1f%% | MAE=%.4f | RMSE=%.4f",
            order,
            clf["accuracy"] * 100,
            reg["mae"],
            reg["rmse"],
        )
        logger.info("Model saved -> %s", model_path)
        logger.info("MLflow run logged to experiment '%s'", mlflow_cfg.experiment_name)


def main() -> None:
    """CLI entrypoint for model training."""
    parser = argparse.ArgumentParser(description="Train an ARIMA model and log to MLflow")
    parser.add_argument("--data-path", type=Path, default=DATA_PATH)
    parser.add_argument("--model-dir", type=Path, default=MODELS_DIR)
    parser.add_argument("--arima-p", type=int, default=1, help="AR order")
    parser.add_argument("--arima-d", type=int, default=0, help="Differencing order")
    parser.add_argument("--arima-q", type=int, default=1, help="MA order")
    parser.add_argument("--test-split", type=float, default=0.2, help="Fraction held out for evaluation")
    parser.add_argument("--seed", type=int, default=DEFAULT_CONFIG.training.seed)
    args = parser.parse_args()

    setup_logging()
    set_seed(args.seed)

    train(
        args.data_path,
        args.model_dir,
        args.arima_p,
        args.arima_d,
        args.arima_q,
        args.test_split,
    )
    logger.info("Training complete")


if __name__ == "__main__":
    main()
