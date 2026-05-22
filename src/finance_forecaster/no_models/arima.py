import numpy as np
import pandas as pd
from statsmodels.tsa.arima.model import ARIMA


def generate_rolling_arima_forecasts(
    df: pd.DataFrame, prediction_dates: pd.Index, order: tuple[int, int, int] = (1, 0, 1)
) -> pd.DataFrame:
    """Generate rolling ARIMA forecasts."""
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

        forecasts.append({"date": date, "arima_mean_forecast_percent": forecast_value})

    return pd.DataFrame(forecasts)
