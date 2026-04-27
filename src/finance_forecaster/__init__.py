"""finance_forecaster.

Finance Forecasting ML Model Pipeline -- Predicting Next Day Financial Movements
"""

from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("finance_forecaster")
except PackageNotFoundError:
    __version__ = "0.0.0+unknown"

__author__ = "Finance Forecasters"
__email__ = "financeforecasters.mlops489@gmail.com"

__all__ = ["__version__", "__author__", "__email__"]
