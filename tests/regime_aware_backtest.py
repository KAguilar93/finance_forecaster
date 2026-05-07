

import numpy as np
import pandas as pd

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
    """Run regime-aware backtest combining LSTM + ARIMA + GARCH."""
    
    backtest_df = lstm_predictions.copy()
    backtest_df["date"] = pd.to_datetime(backtest_df["date"])
    
    arima_forecasts = arima_forecasts.copy()
    arima_forecasts["date"] = pd.to_datetime(arima_forecasts["date"])
    
    backtest_df = backtest_df.merge(arima_forecasts, on="date", how="left")
    
    working_df = df.copy()
    working_df["date"] = working_df.index
    working_df["next_day_log_return"] = working_df["log_return"].shift(-1)
    
    volatility_cutoff = working_df["garch_conditional_volatility"].quantile(volatility_quantile)
    
    backtest_df = backtest_df.merge(
        working_df[["date", "next_day_log_return", "garch_conditional_volatility",
                    "volatility_regime", "vix_regime", "trend_regime", "market_regime"]],
        on="date", how="left"
    )
    
    signals = []
    positions = []
    
    for _, row in backtest_df.iterrows():
        prob_up = row.get("predicted_probability_up", np.nan)
        arima_mean = row.get("arima_mean_forecast_percent", np.nan)
        regime = row.get("market_regime", "neutral")
        garch_vol = row.get("garch_conditional_volatility", np.nan)
        
        decision = "NO TRADE"
        position = 0.0
        
        if pd.isna(prob_up) or pd.isna(arima_mean) or pd.isna(garch_vol):
            decision = "NO TRADE"
        elif regime == "risk_off" or garch_vol > volatility_cutoff:
            decision = "NO TRADE"
        elif abs(prob_up - 0.5) < signal_threshold:
            decision = "NO TRADE"
        else:
            if regime == "low_vol_uptrend":
                if prob_up > upper_prob and arima_mean > -0.05:
                    decision = "UP"
                elif prob_up < lower_prob and arima_mean < 0:
                    decision = "DOWN"
            elif regime == "low_vol_downtrend":
                if prob_up < lower_prob and arima_mean < 0:
                    decision = "DOWN"
                elif prob_up > upper_prob and arima_mean > -0.05:
                    decision = "UP"
            
            if decision in ["UP", "DOWN"]:
                confidence = abs(prob_up - 0.5) * 2
                if confidence < 0.20:
                    decision = "NO TRADE"
                else:
                    position = min(confidence * 2, 1.0)
                    if decision == "DOWN":
                        position *= -1
        
        signals.append(decision)
        positions.append(position)
    
    backtest_df["regime_aware_signal"] = signals
    backtest_df["position"] = positions
    backtest_df["strategy_log_return"] = backtest_df["position"] * backtest_df["next_day_log_return"]
    
    # Equity curves
    backtest_df["strategy_equity"] = np.exp(backtest_df["strategy_log_return"].fillna(0).cumsum())
    backtest_df["buy_hold_equity"] = np.exp(backtest_df["next_day_log_return"].fillna(0).cumsum())
    
    # Summary metrics (you can expand this)
    print(f"Backtest completed for {ticker}. Trades: {len(backtest_df[backtest_df['position'] != 0])}")
    
    return backtest_df, pd.DataFrame()  # summary_df placeholder
