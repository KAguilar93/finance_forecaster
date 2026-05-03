

 
# ------------------------------------------------------------
# Libraries
# ------------------------------------------------------------

import os
import warnings
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yfinance as yf
from arch import arch_model
from sklearn.metrics import (accuracy_score,classification_report,confusion_matrix,roc_auc_score)
from sklearn.preprocessing import StandardScaler
from sklearn.utils.class_weight import compute_class_weight
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tools.sm_exceptions import InterpolationWarning
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.stattools import adfuller, kpss
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.models import Sequential

try:
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
except NameError:
    BASE_DIR = os.getcwd()
    
RAW_DIR = os.path.join(BASE_DIR, "data", "raw")
PROCESSED_DIR = os.path.join(BASE_DIR, "data", "processed")
REPORTS_DIR = os.path.join(BASE_DIR, "reports")
FIGURES_DIR = os.path.join(REPORTS_DIR, "figures")

def download_stock_data(ticker: str, start_date: str, end_date: str) -> pd.DataFrame:
    """Download adjusted close price and volume data."""
    
    data = yf.download(
        ticker,
        start=start_date,
        end=end_date,
        auto_adjust=True,
        progress=False,
    )
    
    if data.empty:
        raise ValueError(f"No data found for ticker: {ticker}")
        
    if isinstance(data.columns, pd.MultiIndex):
        price = data["Close"]
        volume = data["Volume"]
        
        if isinstance(price, pd.DataFrame):
            price = price.iloc[:, 0]
        
        if isinstance(volume, pd.DataFrame):
            volume = volume.iloc[:, 0]
    else:
        price = data["Close"]
        volume = data["Volume"]
    
    df = pd.DataFrame(
        {
            "price": price,
            "volume": volume,
        }
    )
    
    df.dropna(inplace=True)
    
    return df


def download_external_market_features(start_date: str, end_date: str) -> pd.DataFrame:
    """Download external market, macro, breadth, and sector features."""
    tickers = {
        "vix": "^VIX", 
        "spy": "SPY", 
        "iwm": "IWM", 
        "xlk": "XLK", 
        "xlf": "XLF", 
        "xli": "XLI", 
        "xly": "XLY", 
        "xlp": "XLP", 
        "tlt": "TLT", 
        "hyg": "HYG", 
        "uup": "UUP", 
        "gld": "GLD", 
        "tnx": "^TNX",
    }
    
    feature_data = pd.DataFrame()
    
    for name, ticker in tickers.items():
        data = yf.download(
            ticker,
            start=start_date,
            end=end_date,
            auto_adjust=True,
            progress=False,
        )
        
        if data.empty:
            print(f"Warning: No data found for {ticker}")
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
    
    feature_data["iwm_spy_relative_strength"] = (
        feature_data["iwm_close"] / feature_data["spy_close"]
    )
    
    feature_data["qqq_spy_market_strength"] = (
        feature_data["xlk_close"] / feature_data["spy_close"]
    )
    
    feature_data["tech_vs_staples"] = (
        feature_data["xlk_close"] / feature_data["xlp_close"]
    )
    
    feature_data["discretionary_vs_staples"] = (
        feature_data["xly_close"] / feature_data["xlp_close"]
    )
    
    feature_data["financials_vs_market"] = (
        feature_data["xlf_close"] / feature_data["spy_close"]
    )
    
    feature_data["industrials_vs_market"] = (
        feature_data["xli_close"] / feature_data["spy_close"]
    )
   
    feature_data["credit_risk_appetite"] = (
        feature_data["hyg_close"] / feature_data["tlt_close"]
    )
    
    feature_data["gold_vs_market"] = (
        feature_data["gld_close"] / feature_data["spy_close"]
    )
    
    return feature_data


def add_log_returns(df: pd.DataFrame) -> pd.DataFrame:
    """Add log return features."""
    
    df = df.copy()
    df["log_price"] = np.log(df["price"])
    df["log_return"] = df["log_price"].diff()
    df["log_return_squared"] = df["log_return"] ** 2
    
    return df


def run_adf_test(series: pd.Series, series_name: str) -> dict:
    """Run ADF test."""
    
    result = adfuller(series.dropna())
    
    return {
        "series": series_name,
        "test": "ADF",
        "statistic": result[0],
        "p_value": result[1],
        "lags_used": result[2],
        "observations": result[3],
        "stationary_result": "Stationary" if result[1] <= 0.05 else "Non-stationary",
    }


def run_kpss_test(series: pd.Series, series_name: str) -> dict:
    """Run KPSS test."""
    
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", InterpolationWarning)
        statistic, p_value, lags, critical_values = kpss(
            series.dropna(),
            regression="c",
            nlags="auto",
        )
        
    return {
        "series": series_name,
        "test": "KPSS",
        "statistic": statistic,
        "p_value": p_value,
        "lags_used": lags,
        "observations": len(series.dropna()),
        "stationary_result": "Non-stationary" if p_value <= 0.05 else "Stationary",
    }


def compute_expanding_garch_volatility(returns: pd.Series, min_window: int = 500) -> pd.Series:
    """Compute GARCH(1,1) conditional volatility with expanding window.
    """
    
    print("Computing expanding-window GARCH volatility (this takes ~1-3 minutes)...")
    
    returns_pct = returns.dropna() * 100
    conditional_vol = pd.Series(index=returns_pct.index, dtype=float, name="garch_conditional_volatility")
    
    for i in range(min_window, len(returns_pct)):
        window_returns = returns_pct.iloc[:i + 1]
        
        try:
            am = arch_model(
                window_returns,
                mean="Constant",
                vol="GARCH",
                p=1,
                q=1,
                dist="normal",
                rescale=True,         
            )
            
            res = am.fit(disp="off", show_warning=False)
            conditional_vol.iloc[i] = res.conditional_volatility.iloc[-1]
            
        except Exception:
           
            conditional_vol.iloc[i] = (
                conditional_vol.iloc[i - 1]
                if i > 0 and not pd.isna(conditional_vol.iloc[i - 1])
                else window_returns.std()
            )
    
    print("Expanding GARCH volatility computed.")
    
    return conditional_vol



def fit_garch_model(df: pd.DataFrame, ticker: str, min_window: int = 500) -> tuple[pd.DataFrame, object]:
    """Updated: Fit GARCH(1,1) using expanding window; no look-ahead bias."""
    
    df = df.copy()
    
    df["garch_conditional_volatility"] = compute_expanding_garch_volatility(
        df["log_return"], min_window=min_window
    )
    
   
    df["garch_volatility_change"] = df["garch_conditional_volatility"].diff()
    df["garch_volatility_lag_1"] = df["garch_conditional_volatility"].shift(1)
    df["garch_volatility_lag_2"] = df["garch_conditional_volatility"].shift(2)
    
   
    try:
        recent_returns = (df["log_return"].dropna() * 100).iloc[-800:]
        
        final_model = arch_model(
            recent_returns, mean="Constant", vol="GARCH", p=1, q=1, dist="normal", rescale=True
        )
        
        results = final_model.fit(disp="off")
        
        summary_path = os.path.join(REPORTS_DIR, f"{ticker}_garch_summary.txt")
        
        with open(summary_path, "w", encoding="utf-8") as f:
            f.write(str(results.summary()))
            
        print(f"Final GARCH summary saved: {summary_path}")
    
    except Exception as e:
        print(f"Could not save final GARCH summary: {e}")
    
    return df, None  


def add_lstm_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add technical and volatility features for LSTM."""
    
    df = df.copy()
    df["return_lag_1"] = df["log_return"].shift(1)
    df["return_lag_2"] = df["log_return"].shift(2)
    df["return_lag_3"] = df["log_return"].shift(3)
    df["return_lag_5"] = df["log_return"].shift(5)
    df["momentum_3"] = df["log_return"].rolling(3).sum()
    df["momentum_5"] = df["log_return"].rolling(5).sum()
    df["momentum_10"] = df["log_return"].rolling(10).sum()
    df["rolling_mean_5"] = df["log_return"].rolling(5).mean()
    df["rolling_std_5"] = df["log_return"].rolling(5).std()
    df["rolling_mean_20"] = df["log_return"].rolling(20).mean()
    df["rolling_std_20"] = df["log_return"].rolling(20).std()
    df["volatility_ratio"] = df["rolling_std_5"] / df["rolling_std_20"]
   
    df["volume_momentum_5"] = df["volume"].rolling(5).mean() if "volume" in df.columns else 0
    df["vix_change"] = df["vix_close"].diff()
    df["vix_lag_1"] = df["vix_close"].shift(1)
   
    df["spy_momentum_5"] = df["spy_return"].rolling(5).sum()
    df["iwm_momentum_5"] = df["iwm_return"].rolling(5).sum()
    df["xlk_momentum_5"] = df["xlk_return"].rolling(5).sum()
   
    df["rate_change"] = df["tnx_close"].diff()
    df["dollar_momentum_5"] = df["uup_return"].rolling(5).sum()
    df["bond_momentum_5"] = df["tlt_return"].rolling(5).sum()
    df["credit_momentum_5"] = df["hyg_return"].rolling(5).sum()
    df["target"] = (df["log_return"].shift(-1) > 0).astype(int)
    
    return df


def add_market_regimes(df: pd.DataFrame) -> pd.DataFrame:
    """Add market regime labels based on volatility, VIX, and trend."""
    
    df = df.copy()
    
    # -----------------------------
    # Volatility regime
    # -----------------------------
    
    vol_series = df["garch_conditional_volatility"].dropna()
    
    expanding_vol_cutoff = vol_series.expanding(min_periods=252).quantile(0.75)
    df["vol_cutoff"] = expanding_vol_cutoff.reindex(df.index, method="ffill")
    
    df["volatility_regime"] = np.where(
        df["garch_conditional_volatility"] > df["vol_cutoff"],
        "high_vol",
        "low_vol",
    )
    
    # -----------------------------
    # VIX regime
    # -----------------------------
    
    if "vix_close" in df.columns:
        df["vix_regime"] = np.where(
            df["vix_close"] > 25,
            "high_vix",
            "normal_vix",
        )
        
    else:
        df["vix_regime"] = "unknown_vix"
        
    # -----------------------------
    # Trend regime
    # -----------------------------
    
    df["sma_20"] = df["price"].rolling(20).mean()
    df["sma_50"] = df["price"].rolling(50).mean()
    
    df["trend_regime"] = np.where(
        df["sma_20"] > df["sma_50"],
        "uptrend",
        "downtrend",
    )
    
    # -----------------------------
    # Combined regime
    # -----------------------------
    
    conditions = [
        (df["volatility_regime"] == "high_vol") | (df["vix_regime"] == "high_vix"),
        (df["volatility_regime"] == "low_vol") & (df["trend_regime"] == "uptrend"),
        (df["volatility_regime"] == "low_vol") & (df["trend_regime"] == "downtrend"),
    ]
    
    choices = [
        "risk_off",
        "low_vol_uptrend",
        "low_vol_downtrend",
    ]
    
    df["market_regime"] = np.select(
        conditions,
        choices,
        default="neutral",
    )
    
    return df


def create_lstm_dataset(
    df: pd.DataFrame,
    lookback: int = 30,
) -> tuple:
    """Create train/test LSTM sequences."""
    
    model_df = add_lstm_features(df)
    
    feature_cols = [
        "log_return",
        "log_return_squared",
        "garch_conditional_volatility",
        "garch_volatility_change",
        "garch_volatility_lag_1",
        "garch_volatility_lag_2",
        "return_lag_1",
        "return_lag_2",
        "return_lag_3",
        "return_lag_5",
        "momentum_3",
        "momentum_5",
        "momentum_10",
        "rolling_mean_5",
        "rolling_std_5",
        "rolling_mean_20",
        "rolling_std_20",
        "volatility_ratio",
   
        "vix_close",
        "vix_return",
        "vix_change",
        "vix_lag_1",
   
        "spy_return",
        "spy_momentum_5",
   
        "iwm_return",
        "iwm_momentum_5",
        "iwm_spy_relative_strength",
        "credit_risk_appetite",
   
        "xlk_return",
        "xlk_momentum_5",
        "qqq_spy_market_strength",
        "tech_vs_staples",
        "discretionary_vs_staples",
        "financials_vs_market",
        "industrials_vs_market",
   
        "tnx_close",
        "rate_change",
        "uup_return",
        "dollar_momentum_5",
        "tlt_return",
        "bond_momentum_5",
        "hyg_return",
        "credit_momentum_5",
        "gld_return",
        "gold_vs_market",
    ]
    
    model_df.dropna(inplace=True)
    
    features = model_df[feature_cols].values
    target = model_df["target"].values
    dates = model_df.index
    
    split_index = int(len(features) * 0.8)
    scaler = StandardScaler()
    
    features_train_scaled = scaler.fit_transform(features[:split_index])
    features_test_scaled = scaler.transform(features[split_index:])
    features_scaled = np.vstack([features_train_scaled, features_test_scaled])
    
    X = []
    y = []
    
    sequence_dates = []
    
    for i in range(lookback, len(features_scaled)):
        X.append(features_scaled[i - lookback:i])
        y.append(target[i])
        sequence_dates.append(dates[i])
        
    X = np.array(X)
    y = np.array(y)
    
    sequence_dates = np.array(sequence_dates)
    train_sequence_count = split_index - lookback
    
    X_train = X[:train_sequence_count]
    X_test = X[train_sequence_count:]
    y_train = y[:train_sequence_count]
    y_test = y[train_sequence_count:]
    test_dates = sequence_dates[train_sequence_count:]
    
    return X_train, X_test, y_train, y_test, test_dates, scaler, feature_cols, model_df


def train_lstm_garch_model(
    df: pd.DataFrame,
    ticker: str,
    lookback: int = 30,
    epochs: int = 100,
    batch_size: int = 32,
    prediction_threshold: float = 0.55,
) -> tuple:
    """Train LSTM using GARCH and return-based features."""
    (
        X_train,
        X_test,
        y_train,
        y_test,
        test_dates,
        scaler,
        feature_cols,
        model_df,
    ) = create_lstm_dataset(df, lookback=lookback)
    
    classes = np.unique(y_train)
    weights = compute_class_weight(
        class_weight="balanced",
        classes=classes,
        y=y_train,
    )
    
    class_weights = dict(zip(classes, weights))
    model = Sequential()
    
    model.add(
        LSTM(
            units=64,
            return_sequences=True,
            input_shape=(X_train.shape[1], X_train.shape[2]),
        )
    )
    
    model.add(Dropout(0.30))
    model.add(LSTM(units=32))
    model.add(Dropout(0.30))
    model.add(Dense(16, activation="relu"))
    model.add(Dense(1, activation="sigmoid"))
    
    model.compile(
        optimizer="adam",
        loss="binary_crossentropy",
        metrics=["accuracy"],
    )
    
    early_stop = EarlyStopping(
        monitor="val_loss",
        patience=10,
        restore_best_weights=True,
    )
    
    history = model.fit(
        X_train,
        y_train,
        validation_split=0.2,
        epochs=epochs,
        batch_size=batch_size,
        callbacks=[early_stop],
        class_weight=class_weights,
        verbose=1,
    )
    
    y_pred_prob = model.predict(X_test).flatten()
    y_pred = (y_pred_prob >= prediction_threshold).astype(int)
    
    accuracy = accuracy_score(y_test, y_pred)
    majority_class = int(pd.Series(y_train).mode()[0])
    
    baseline_pred = np.full_like(y_test, majority_class)
    baseline_accuracy = accuracy_score(y_test, baseline_pred)
    
    try:
        roc_auc = roc_auc_score(y_test, y_pred_prob)
    except ValueError:
        roc_auc = np.nan
        
    report = classification_report(y_test, y_pred, zero_division=0)
    matrix = confusion_matrix(y_test, y_pred)
    
    print("\n--- LSTM + GARCH Direction Model Results ---")
    print(f"LSTM Accuracy: {accuracy:.4f}")
    print(f"Baseline Accuracy: {baseline_accuracy:.4f}")
    print(f"ROC AUC: {roc_auc:.4f}")
    print(f"Prediction Threshold: {prediction_threshold}")
    print("\nClassification Report:")
    print(report)
    print("\nConfusion Matrix:")
    print(matrix)
    
    model_path = os.path.join(REPORTS_DIR, f"{ticker}_lstm_garch_direction_model.keras")
    model.save(model_path)
    
    history_df = pd.DataFrame(history.history)
    history_path = os.path.join(REPORTS_DIR, f"{ticker}_lstm_garch_training_history.csv")
    history_df.to_csv(history_path, index=False)
    
    predictions_df = pd.DataFrame(
        {
            "date": test_dates,
            "actual_direction": y_test,
            "predicted_probability_up": y_pred_prob,
            "predicted_direction": y_pred,
        }
    )
    
    predictions_path = os.path.join(REPORTS_DIR, f"{ticker}_lstm_garch_predictions.csv")
    predictions_df.to_csv(predictions_path, index=False)
    
    metrics_path = os.path.join(REPORTS_DIR, f"{ticker}_lstm_garch_metrics.txt")
    
    with open(metrics_path, "w", encoding="utf-8") as file:
        file.write(f"Ticker: {ticker}\n")
        file.write(f"Lookback window: {lookback}\n")
        file.write(f"Prediction threshold: {prediction_threshold}\n")
        file.write(f"Features: {feature_cols}\n")
        file.write(f"Class weights: {class_weights}\n")
        file.write(f"LSTM Accuracy: {accuracy:.4f}\n")
        file.write(f"Baseline Accuracy: {baseline_accuracy:.4f}\n")
        file.write(f"ROC AUC: {roc_auc:.4f}\n\n")
        file.write("Classification Report:\n")
        file.write(report)
        file.write("\nConfusion Matrix:\n")
        file.write(str(matrix))
        
    save_lstm_training_plots(history_df, ticker)
    
    print(f"\nSaved model: {model_path}")
    print(f"Saved metrics: {metrics_path}")
    print(f"Saved predictions: {predictions_path}")
    print(f"Saved training history: {history_path}")
    
    return model, scaler, feature_cols, predictions_df


def save_lstm_training_plots(history_df: pd.DataFrame, ticker: str) -> None:
    """Save LSTM loss and accuracy plots."""
    
    loss_path = os.path.join(FIGURES_DIR, f"{ticker}_lstm_garch_loss.png")
    accuracy_path = os.path.join(FIGURES_DIR, f"{ticker}_lstm_garch_accuracy.png")
    
    plt.figure(figsize=(10, 5))
    plt.plot(history_df["loss"], label="Train Loss")
    plt.plot(history_df["val_loss"], label="Validation Loss")
    plt.title(f"{ticker} LSTM + GARCH Training Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(loss_path, dpi=300, bbox_inches="tight")
    plt.close()
    
    if "accuracy" in history_df.columns and "val_accuracy" in history_df.columns:
        plt.figure(figsize=(10, 5))
        plt.plot(history_df["accuracy"], label="Train Accuracy")
        plt.plot(history_df["val_accuracy"], label="Validation Accuracy")
        plt.title(f"{ticker} LSTM + GARCH Training Accuracy")
        plt.xlabel("Epoch")
        plt.ylabel("Accuracy")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(accuracy_path, dpi=300, bbox_inches="tight")
        plt.close()
        
        
def forecast_lstm_garch_next_day(
    df: pd.DataFrame,
    lstm_model: object,
    scaler: StandardScaler,
    feature_cols: list[str],
    ticker: str,
    lookback: int = 30,
    threshold: float = 0.55,
) -> pd.DataFrame:
    """Forecast next-day direction using trained LSTM + GARCH features."""
    model_df = add_lstm_features(df)
    model_df.dropna(inplace=True)
    
    features = model_df[feature_cols].values
    features_scaled = scaler.transform(features)
    
    last_sequence = features_scaled[-lookback:]
    last_sequence = np.expand_dims(last_sequence, axis=0)
    
    probability_up = float(lstm_model.predict(last_sequence).flatten()[0])
    
    if probability_up >= threshold:
        direction = "UP"
        
    elif probability_up <= (1 - threshold):
        direction = "DOWN"
        
    else:
        direction = "NO TRADE"
        
    forecast_df = pd.DataFrame(
        {
            "ticker": [ticker],
            "lstm_garch_probability_up": [probability_up],
            "threshold": [threshold],
            "lstm_garch_predicted_direction": [direction],
        }
    )
    
    forecast_path = os.path.join(REPORTS_DIR, f"{ticker}_lstm_garch_next_day_forecast.csv")
    forecast_df.to_csv(forecast_path, index=False)
    
    print("\n--- LSTM + GARCH Next-Day Forecast ---")
    print(f"Ticker: {ticker}")
    print(f"Probability UP: {probability_up:.4f}")
    print(f"Threshold: {threshold}")
    print(f"Predicted direction: {direction}")
    print(f"Saved forecast: {forecast_path}")
    
    return forecast_df


def generate_rolling_arima_forecasts(
    df: pd.DataFrame,
    prediction_dates,
    order: tuple[int, int, int] = (1, 0, 1),
) -> pd.DataFrame:
    """Generate rolling ARIMA one-day-ahead mean forecasts for backtesting."""
    
    returns_percent = df["log_return"].dropna() * 100
    forecasts = []
    
    for date in prediction_dates:
        date = pd.to_datetime(date)
        train_series = returns_percent.loc[:date].dropna()
        
        if len(train_series) < 60:
            forecast_value = np.nan
        
        else:
            try:
                model = ARIMA(train_series, order=order)
                result = model.fit()
                forecast_value = result.forecast(steps=1).iloc[0]
            except Exception:
                forecast_value = np.nan
        
        forecasts.append(
            {
                "date": date,
                "arima_mean_forecast_percent": forecast_value,
            }
        )
    
    return pd.DataFrame(forecasts)


def run_regime_aware_backtest(
    df: pd.DataFrame,
    lstm_predictions: pd.DataFrame,
    arima_forecasts: pd.DataFrame,
    ticker: str,
    upper_prob: float = 0.55,
    lower_prob: float = 0.45,
    signal_threshold: float = 0.08,
    volatility_quantile: float = 0.75,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Backtest combined LSTM + ARIMA + GARCH strategy with regime awareness."""
    
    backtest_df = lstm_predictions.copy()
    backtest_df["date"] = pd.to_datetime(backtest_df["date"])
    
    arima_forecasts = arima_forecasts.copy()
    arima_forecasts["date"] = pd.to_datetime(arima_forecasts["date"])
    
    backtest_df = backtest_df.merge(arima_forecasts, on="date", how="left")
    
    working_df = df.copy()
    working_df["date"] = working_df.index
    working_df["next_day_log_return"] = working_df["log_return"].shift(-1)
    
    volatility_cutoff = working_df["garch_conditional_volatility"].quantile(
        volatility_quantile
    )
    
    backtest_df = backtest_df.merge(
        working_df[
            [
                "date",
                "next_day_log_return",
                "garch_conditional_volatility",
                "volatility_regime",
                "vix_regime",
                "trend_regime",
                "market_regime",
            ]
        ],
        on="date",
        how="left",
    )
    
    signals = []
    positions = []
    
    for _, row in backtest_df.iterrows():
        prob_up = row["predicted_probability_up"]
        arima_mean = row["arima_mean_forecast_percent"]
        regime = row["market_regime"]
        garch_vol = row["garch_conditional_volatility"]
        signal = prob_up - 0.5
        decision = "NO TRADE"
        position = 0
        
        if pd.isna(prob_up) or pd.isna(arima_mean) or pd.isna(garch_vol):
            decision = "NO TRADE"
            position = 0
            
        elif regime == "risk_off":
            decision = "NO TRADE"
            position = 0
            
        elif garch_vol > volatility_cutoff:
            decision = "NO TRADE"
            position = 0
            
        elif abs(signal) < signal_threshold:
            decision = "NO TRADE"
            position = 0
            
        elif regime == "low_vol_uptrend":
            if prob_up > upper_prob and arima_mean > -0.05:
                decision = "UP"
            elif prob_up < lower_prob and arima_mean < 0:
                decision = "DOWN"
            elif prob_up > 0.56:
                decision = "UP"
            elif prob_up < 0.44:
                decision = "DOWN"
        elif regime == "low_vol_downtrend":
            if prob_up < lower_prob and arima_mean < 0:
                decision = "DOWN"
            elif prob_up > 0.65 and arima_mean > -0.05:
                decision = "UP"
            elif prob_up < 0.44:
                decision = "DOWN"
            elif prob_up > 0.56:
                decision = "UP"
        if regime == "low_vol_uptrend" and decision == "DOWN":
            decision = "NO TRADE"
        if regime == "low_vol_downtrend" and decision == "UP":
            decision = "NO TRADE"
            
        if decision in ["UP", "DOWN"]:
            confidence = abs(prob_up - 0.5) * 2
            
            if confidence < 0.20:
                decision = "NO TRADE"
               
            position = min(confidence * 2, 1.0)
            if decision == "DOWN":
                position *= -1
        else:
            position = 0
            
        signals.append(decision)
        positions.append(position)
        
    backtest_df["regime_aware_signal"] = signals
    backtest_df["position"] = positions
    backtest_df["strategy_log_return"] = (
        backtest_df["position"] * backtest_df["next_day_log_return"]
    )
    
    backtest_df["buy_hold_log_return"] = backtest_df["next_day_log_return"]
    backtest_df["strategy_equity"] = np.exp(
        backtest_df["strategy_log_return"].fillna(0).cumsum()
    )
    
    backtest_df["buy_hold_equity"] = np.exp(
        backtest_df["buy_hold_log_return"].fillna(0).cumsum()
    )
    
    traded = backtest_df[backtest_df["position"] != 0].copy()
    
    if len(traded) > 0:
        traded["correct_direction"] = np.where(
            ((traded["position"] > 0) & (traded["next_day_log_return"] > 0))
            | ((traded["position"] < 0) & (traded["next_day_log_return"] < 0)),
            1,
            0,
        )
        
        hit_rate = traded["correct_direction"].mean()
        trade_count = len(traded)
    
    else:
        hit_rate = np.nan
        trade_count = 0
    
    strategy_returns = backtest_df["strategy_log_return"].dropna()
    
    if strategy_returns.std() != 0:
        sharpe_ratio = (
            strategy_returns.mean() / strategy_returns.std()
        ) * np.sqrt(252)
        
    else:
        sharpe_ratio = np.nan
    running_max = backtest_df["strategy_equity"].cummax()
    drawdown = backtest_df["strategy_equity"] / running_max - 1
    
    max_drawdown = drawdown.min()
    
    regime_counts = backtest_df["market_regime"].value_counts().to_dict()
    
    summary_df = pd.DataFrame(
        {
            "ticker": [ticker],
            "upper_probability_threshold": [upper_prob],
            "lower_probability_threshold": [lower_prob],
            "signal_threshold": [signal_threshold],
            "volatility_quantile_cutoff": [volatility_quantile],
            "volatility_cutoff": [volatility_cutoff],
            "total_test_days": [len(backtest_df)],
            "trade_count": [trade_count],
            "trade_rate": [trade_count / len(backtest_df)],
            "hit_rate_on_trades": [hit_rate],
            "strategy_total_return": [backtest_df["strategy_equity"].iloc[-1] - 1],
            "buy_hold_total_return": [backtest_df["buy_hold_equity"].iloc[-1] - 1],
            "annualized_sharpe": [sharpe_ratio],
            "max_drawdown": [max_drawdown],
            "risk_off_days": [regime_counts.get("risk_off", 0)],
            "low_vol_uptrend_days": [regime_counts.get("low_vol_uptrend", 0)],
            "low_vol_downtrend_days": [regime_counts.get("low_vol_downtrend", 0)],
            "neutral_days": [regime_counts.get("neutral", 0)],
            "average_position_size": [traded["position"].abs().mean() if len(traded) > 0 else 0],
        }
    )
    
    backtest_path = os.path.join(REPORTS_DIR, f"{ticker}_regime_aware_backtest.csv")
    summary_path = os.path.join(
        REPORTS_DIR, f"{ticker}_regime_aware_backtest_summary.csv"
    )
    
    backtest_df.to_csv(backtest_path, index=False)
    summary_df.to_csv(summary_path, index=False)
    
    print("\n--- Regime-Aware Backtest Results ---")
    print(summary_df.T)
    print(f"\nSaved regime-aware backtest: {backtest_path}")
    print(f"Saved regime-aware summary: {summary_path}")
    return backtest_df, summary_df


def save_regime_aware_equity_curve(
    backtest_df: pd.DataFrame,
    ticker: str,
) -> None:
    """Save regime-aware equity curve."""
    
    path = os.path.join(FIGURES_DIR, f"{ticker}_regime_aware_equity_curve.png")
    
    plt.figure(figsize=(12, 5))
    plt.plot(backtest_df["date"], backtest_df["strategy_equity"], label="Regime-Aware Strategy")
    plt.plot(backtest_df["date"], backtest_df["buy_hold_equity"], label="Buy and Hold")
    plt.title(f"{ticker} Regime-Aware Backtest Equity Curve")
    plt.xlabel("Date")
    plt.ylabel("Growth of $1")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(path, dpi=300, bbox_inches="tight")
    
    plt.close()
    print(f"Saved regime-aware equity curve: {path}")
    
    
def save_backtest_equity_curve(
    backtest_df: pd.DataFrame,
    ticker: str,
) -> None:
    """Save equity curve comparing strategy vs buy-and-hold."""
    
    path = os.path.join(FIGURES_DIR, f"{ticker}_combined_backtest_equity_curve.png")
    
    plt.figure(figsize=(12, 5))
    plt.plot(backtest_df["date"], backtest_df["strategy_equity"], label="Strategy")
    plt.plot(backtest_df["date"], backtest_df["buy_hold_equity"], label="Buy and Hold")
    plt.title(f"{ticker} Combined Model Backtest Equity Curve")
    plt.xlabel("Date")
    plt.ylabel("Growth of $1")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(path, dpi=300, bbox_inches="tight")
    plt.close()
    
    print(f"Saved backtest equity curve: {path}")
    
    
def save_time_series_plot(
    df: pd.DataFrame,
    column: str,
    title: str,
    y_label: str,
    path: str,
) -> None:
    """Save time series plot."""
    
    plt.figure(figsize=(12, 5))
    plt.plot(df.index, df[column])
    plt.title(title)
    plt.xlabel("Date")
    plt.ylabel(y_label)
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(path, dpi=300, bbox_inches="tight")
    plt.close()
    
    
def save_acf_plot(series: pd.Series, title: str, path: str, lags: int = 40) -> None:
    """Save ACF plot."""
    
    fig, ax = plt.subplots(figsize=(10, 5))
    plot_acf(series.dropna(), lags=lags, ax=ax)
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    
    
def save_pacf_plot(series: pd.Series, title: str, path: str, lags: int = 40) -> None:
    """Save PACF plot."""
    
    fig, ax = plt.subplots(figsize=(10, 5))
    plot_pacf(series.dropna(), lags=lags, method="ywm", ax=ax)
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    
    
def save_garch_volatility_plot(df: pd.DataFrame, ticker: str, path: str) -> None:
    """Save GARCH conditional volatility plot."""
    
    plt.figure(figsize=(12, 5))
    plt.plot(df.index, df["garch_conditional_volatility"])
    plt.title(f"{ticker} GARCH(1,1) Conditional Volatility")
    plt.xlabel("Date")
    plt.ylabel("Conditional Volatility (%)")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(path, dpi=300, bbox_inches="tight")
    plt.close()
    

def add_trade_recommendation_column(
    forecast_df: pd.DataFrame,
    full_df: pd.DataFrame,
    upper_prob: float = 0.53,
    lower_prob: float = 0.47,
    volatility_quantile: float = 0.75,
) -> pd.DataFrame:
    """Add trade recommendation column using probability and volatility filter."""

    vol_cutoff = full_df["garch_conditional_volatility"].quantile(volatility_quantile)
    current_vol = full_df["garch_conditional_volatility"].iloc[-1]

    recommendations = []

    for _, row in forecast_df.iterrows():
        prob = row["lstm_garch_probability_up"]

        if current_vol > vol_cutoff:
            recommendations.append("NO TRADE")
        elif prob > upper_prob:
            recommendations.append("TRADE (UP)")
        elif prob < lower_prob:
            recommendations.append("TRADE (DOWN)")
        else:
            recommendations.append("NO TRADE")

    forecast_df["trade_recommendation"] = recommendations

    return forecast_df
    
    
def run_pipeline(ticker: str, start_date: str, end_date: str) -> None:
    """Run full pipeline."""
    
    ticker = ticker.upper()
    
    os.makedirs(RAW_DIR, exist_ok=True)
    os.makedirs(PROCESSED_DIR, exist_ok=True)
    os.makedirs(REPORTS_DIR, exist_ok=True)
    os.makedirs(FIGURES_DIR, exist_ok=True)
    
    print("=" * 60)
    print(f"Saving outputs under:\n{BASE_DIR}")
    print("=" * 60)
    
    df = download_stock_data(ticker, start_date, end_date)
    df = add_log_returns(df)
    external_features = download_external_market_features(start_date, end_date)
    
    df = df.merge(
        external_features,
        left_index=True,
        right_index=True,
        how="left",
        )
    
    df.ffill(inplace=True)
    
    df.dropna(inplace=True)
    
    raw_path = os.path.join(RAW_DIR, f"{ticker}_raw.csv")
    processed_path = os.path.join(PROCESSED_DIR, f"{ticker}_processed.csv")
    
    df[["price"]].to_csv(raw_path)
    stationarity_results = [
        run_adf_test(df["price"], "price"),
        run_kpss_test(df["price"], "price"),
        run_adf_test(df["log_return"], "log_return"),
        run_kpss_test(df["log_return"], "log_return"),
    ]
    
    stationarity_df = pd.DataFrame(stationarity_results)
    stationarity_df.insert(0, "ticker", ticker)
    stationarity_path = os.path.join(REPORTS_DIR, f"{ticker}_stationarity_results.csv")
    stationarity_df.to_csv(stationarity_path, index=False)
    
    df, garch_results = fit_garch_model(df, ticker, min_window=500)  
   
    df = add_market_regimes(df)
    
    lstm_model, scaler, feature_cols, predictions_df = train_lstm_garch_model(
        df=df,
        ticker=ticker,
        lookback=30,
        epochs=100,
        batch_size=32,
        prediction_threshold=0.55,
    )
    
    # ------------------------------------------------------------
    # LSTM next-day forecast
    # ------------------------------------------------------------
    lstm_forecast_df = forecast_lstm_garch_next_day(
        df=df,
        lstm_model=lstm_model,
        scaler=scaler,
        feature_cols=feature_cols,
        ticker=ticker,
        lookback=30,
        threshold=0.55,
    )

    # ------------------------------------------------------------
    # Add trade recommendation column to forecast report
    # ------------------------------------------------------------
    lstm_forecast_df = add_trade_recommendation_column(
        forecast_df=lstm_forecast_df,
        full_df=df,
        upper_prob=0.53,
        lower_prob=0.47,
        volatility_quantile=0.75,
    )

    print("\n--- FINAL FORECAST WITH TRADE RECOMMENDATION ---")
    print(lstm_forecast_df)

    forecast_trade_path = os.path.join(
        REPORTS_DIR,
        f"{ticker}_lstm_garch_forecast_with_trade_recommendation.csv",
    )

    lstm_forecast_df.to_csv(forecast_trade_path, index=False)

    print(f"Saved forecast with trade recommendation: {forecast_trade_path}")
   
    print("\nGenerating rolling ARIMA forecasts for backtest...")
    
    arima_forecasts = generate_rolling_arima_forecasts(
        df=df,
        prediction_dates=predictions_df["date"],
        order=(1, 0, 1),
    )
   
    print("\nRunning combined LSTM + ARIMA + GARCH backtest...")
    backtest_df, backtest_summary = run_regime_aware_backtest(
        df=df,
        lstm_predictions=predictions_df,
        arima_forecasts=arima_forecasts,
        ticker=ticker,
        upper_prob=0.53,
        lower_prob=0.47,
        signal_threshold=0.03,
        volatility_quantile=0.75,
    )
   
    save_regime_aware_equity_curve(backtest_df, ticker)
    
    df.to_csv(processed_path)
    
    save_time_series_plot(
        df=df,
        column="price",
        title=f"{ticker} Adjusted Closing Price",
        y_label="Adjusted Price",
        path=os.path.join(FIGURES_DIR, f"{ticker}_price.png"),
    )
    
    save_time_series_plot(
        df=df,
        column="log_return",
        title=f"{ticker} Log Returns",
        y_label="Log Return",
        path=os.path.join(FIGURES_DIR, f"{ticker}_log_returns.png"),
    )
    
    save_acf_plot(
        series=df["log_return"],
        title=f"{ticker} Log Returns ACF",
        path=os.path.join(FIGURES_DIR, f"{ticker}_log_returns_acf.png"),
    )
    
    save_pacf_plot(
        series=df["log_return"],
        title=f"{ticker} Log Returns PACF",
        path=os.path.join(FIGURES_DIR, f"{ticker}_log_returns_pacf.png"),
    )
    
    save_time_series_plot(
        df=df,
        column="log_return_squared",
        title=f"{ticker} Squared Log Returns",
        y_label="Squared Log Return",
        path=os.path.join(FIGURES_DIR, f"{ticker}_squared_log_returns.png"),
    )
    
    save_acf_plot(
        series=df["log_return_squared"],
        title=f"{ticker} Squared Log Returns ACF",
        path=os.path.join(FIGURES_DIR, f"{ticker}_squared_log_returns_acf.png"),
    )
    
    save_garch_volatility_plot(
        df=df,
        ticker=ticker,
        path=os.path.join(FIGURES_DIR, f"{ticker}_garch_conditional_volatility.png"),
    )
    
    print("\nPipeline complete.")
    print(f"Reports saved here:\n{REPORTS_DIR}")
    print(f"Figures saved here:\n{FIGURES_DIR}")
    
    
if __name__ == "__main__":
    ticker = "qqq"
    start = "2015-01-01"
    end = "2025-01-01"
    run_pipeline(ticker, start, end)
