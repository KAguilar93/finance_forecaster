

import pandas as pd
import numpy as np
from .config import REPORTS_DIR, FIGURES_DIR, PROCESSED_DIR
from .data.download import download_stock_data, download_external_market_features
from .data.features import add_log_returns, add_lstm_features
from .models.garch import fit_garch_model
from .models.stationarity import run_adf_test, run_kpss_test
from .models.arima import generate_rolling_arima_forecasts
from .models.lstm import train_lstm_model
from .backtest.regime_aware import run_regime_aware_backtest
from .utils.plotting import save_regime_aware_equity_curve


def run_pipeline(ticker: str = "qqq", start_date: str = "2015-01-01", end_date: str = "2025-01-01"):
    """Full pipeline with real LSTM training."""
    print(f"Starting full pipeline for {ticker}...")

    # Data loading and features
    df = download_stock_data(ticker, start_date, end_date)
    df = add_log_returns(df)
    external = download_external_market_features(start_date, end_date)
    df = df.merge(external, left_index=True, right_index=True, how="left")
    df.ffill(inplace=True)
    df.dropna(inplace=True)

    # Stationarity tests
    stationarity_results = [
        run_adf_test(df["price"], "price"),
        run_kpss_test(df["price"], "price"),
        run_adf_test(df["log_return"], "log_return"),
        run_kpss_test(df["log_return"], "log_return"),
    ]
    pd.DataFrame(stationarity_results).to_csv(f"{REPORTS_DIR}/{ticker}_stationarity.csv", index=False)

    # GARCH
    df = fit_garch_model(df, ticker)

    # LSTM features + training
    df = add_lstm_features(df)

    # Prepare data for LSTM
    feature_cols = [col for col in df.columns if col not in ['target']]
    features = df[feature_cols].values
    target = df["target"].values

    split_index = int(len(features) * 0.8)
    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(features)

    lookback = 30
    X, y = [], []
    for i in range(lookback, len(features_scaled)):
        X.append(features_scaled[i-lookback:i])
        y.append(target[i])
    X = np.array(X)
    y = np.array(y)

    X_train = X[:split_index]
    X_test = X[split_index:]
    y_train = y[:split_index]
    y_test = y[split_index:]

    # Full LSTM training
    model, history, y_pred_prob, y_pred = train_lstm_model(X_train, y_train, X_test, y_test)

    # ARIMA
    prediction_dates = df.index[-len(y_test):]
    arima_forecasts = generate_rolling_arima_forecasts(df, prediction_dates)

    # Backtest
    lstm_predictions = pd.DataFrame({
        "date": prediction_dates,
        "predicted_probability_up": y_pred_prob
    })
    backtest_df, summary = run_regime_aware_backtest(df, lstm_predictions, arima_forecasts, ticker)

    # Save outputs
    df.to_csv(f"{PROCESSED_DIR}/{ticker}_processed.csv", index=True)
    if not backtest_df.empty:
        backtest_df.to_csv(f"{REPORTS_DIR}/{ticker}_backtest.csv", index=False)

    print(f"\nFull pipeline completed for {ticker}!")
    print(f"Reports saved in: {REPORTS_DIR}")


if __name__ == "__main__":
    run_pipeline()
