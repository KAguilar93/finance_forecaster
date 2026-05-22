from __future__ import annotations

import warnings
from typing import Any

import pandas as pd
from statsmodels.tools.sm_exceptions import InterpolationWarning
from statsmodels.tsa.stattools import adfuller, kpss


def run_adf_test(series: pd.Series, series_name: str) -> dict[str, Any]:
    result = adfuller(series.dropna())
    return {
        "series": series_name,
        "test": "ADF",
        "statistic": result[0],
        "p_value": result[1],
        "stationary": result[1] <= 0.05,
    }


def run_kpss_test(series: pd.Series, series_name: str) -> dict[str, Any]:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", InterpolationWarning)
        statistic, p_value, lags, critical = kpss(series.dropna(), regression="c", nlags="auto")
    return {
        "series": series_name,
        "test": "KPSS",
        "statistic": statistic,
        "p_value": p_value,
        "stationary": p_value > 0.05,
    }
