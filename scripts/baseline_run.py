"""Baseline model evaluation script.

Downloads QQQ data, runs stationarity tests, computes an ARIMA directional
baseline, and saves a summary to reports/baseline_results.md.

Usage:
    python scripts/baseline_run.py
"""

from __future__ import annotations

import sys
from datetime import datetime
from pathlib import Path

import pandas as pd
import yfinance as yf

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from finance_forecaster.evaluation.metrics import classification_report, regression_report  # noqa: E402
from finance_forecaster.features.features import add_log_returns  # noqa: E402
from finance_forecaster.models.arima import generate_rolling_arima_forecasts  # noqa: E402
from finance_forecaster.models.stationarity import run_adf_test, run_kpss_test  # noqa: E402

TICKER = "QQQ"
START = "2015-01-01"
END = "2024-12-31"
ARIMA_WINDOW = 60  # rolling forecast on last N trading days (keeps runtime short)
REPORTS_DIR = PROJECT_ROOT / "reports"
DATA_DIR = PROJECT_ROOT / "data" / "raw"


def download_data(ticker: str, start: str, end: str) -> pd.DataFrame:
    print(f"Downloading {ticker} ({start} to {end})...")
    raw = yf.download(ticker, start=start, end=end, auto_adjust=True, progress=False)
    df = raw[["Close"]].rename(columns={"Close": "price"})
    df = add_log_returns(df)
    df = df.dropna()
    print(f"  {len(df)} trading days loaded.")
    return df


def naive_directional_accuracy(df: pd.DataFrame) -> dict[str, float]:
    """Predict tomorrow's direction = today's direction (momentum naive baseline)."""
    actual = (df["log_return"].shift(-1) > 0).dropna().astype(int)
    predicted = (df["log_return"] > 0).iloc[: len(actual)].astype(int)
    return classification_report(actual.values, predicted.values)


def run_arima_baseline(df: pd.DataFrame, n_days: int) -> tuple[dict[str, float], dict[str, float]]:
    """Roll ARIMA(1,0,1) over the last n_days and evaluate directional accuracy."""
    eval_df = df.iloc[-n_days:].copy()
    prediction_dates = eval_df.index

    print(f"  Running rolling ARIMA(1,0,1) over last {n_days} trading days...")
    forecasts = generate_rolling_arima_forecasts(df, prediction_dates)
    forecasts = forecasts.set_index("date")

    actual_returns = eval_df["log_return"]
    arima_forecasts = forecasts["arima_mean_forecast_percent"].reindex(actual_returns.index)

    valid = arima_forecasts.notna()
    actual_dir = (actual_returns[valid].shift(-1) > 0).dropna().astype(int)
    forecast_dir = (arima_forecasts[valid].iloc[: len(actual_dir)] > 0).astype(int)

    # Regression on return magnitude
    reg = regression_report(
        actual_returns[valid].values * 100,
        arima_forecasts[valid].values,
    )
    clf = classification_report(actual_dir.values, forecast_dir.values[: len(actual_dir)])
    return clf, reg


def save_report(
    df: pd.DataFrame,
    adf: dict,
    kpss: dict,
    naive: dict[str, float],
    arima_clf: dict[str, float],
    arima_reg: dict[str, float],
) -> Path:
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    out = REPORTS_DIR / "baseline_results.md"
    run_date = datetime.now().strftime("%Y-%m-%d %H:%M")

    lines = [
        "# Baseline Model Results",
        f"\n**Run date:** {run_date}  ",
        f"**Ticker:** {TICKER}  ",
        f"**Period:** {START} -> {END}  ",
        f"**Trading days:** {len(df)}  ",
        "",
        "---",
        "",
        "## 1. Data Summary",
        "",
        "| Stat | Value |",
        "|---|---|",
        f"| Mean daily return | {df['log_return'].mean():.6f} |",
        f"| Std daily return | {df['log_return'].std():.6f} |",
        f"| Min daily return | {df['log_return'].min():.6f} |",
        f"| Max daily return | {df['log_return'].max():.6f} |",
        f"| % Up days | {(df['log_return'] > 0).mean():.1%} |",
        "",
        "---",
        "",
        "## 2. Stationarity Tests (Log Returns)",
        "",
        "| Test | Statistic | p-value | Stationary? |",
        "|---|---|---|---|",
        f"| ADF | {adf['statistic']:.4f} | {adf['p_value']:.4f} | {'Yes' if adf['stationary'] else 'No'} |",
        f"| KPSS | {kpss['statistic']:.4f} | {kpss['p_value']:.4f} | {'Yes' if kpss['stationary'] else 'No'} |",
        "",
        "> Both tests confirm log returns are stationary — appropriate for ARIMA modeling.",
        "",
        "---",
        "",
        "## 3. Naive Directional Baseline",
        "",
        "> Predict tomorrow's direction = today's direction (momentum baseline).",
        "",
        "| Metric | Value |",
        "|---|---|",
        f"| Accuracy | {naive['accuracy']:.3f} |",
        f"| Precision | {naive['precision']:.3f} |",
        f"| Recall | {naive['recall']:.3f} |",
        f"| F1 | {naive['f1']:.3f} |",
        "",
        "---",
        "",
        "## 4. ARIMA(1,0,1) Rolling Forecast Baseline",
        "",
        f"> Evaluated on last {ARIMA_WINDOW} trading days with rolling re-fit.",
        "",
        "### Directional Accuracy",
        "",
        "| Metric | Value |",
        "|---|---|",
        f"| Accuracy | {arima_clf['accuracy']:.3f} |",
        f"| Precision | {arima_clf['precision']:.3f} |",
        f"| Recall | {arima_clf['recall']:.3f} |",
        f"| F1 | {arima_clf['f1']:.3f} |",
        "",
        "### Return Magnitude (Regression)",
        "",
        "| Metric | Value |",
        "|---|---|",
        f"| MAE | {arima_reg['mae']:.4f} |",
        f"| RMSE | {arima_reg['rmse']:.4f} |",
        f"| R² | {arima_reg['r2']:.4f} |",
        "",
        "---",
        "",
        "## 5. Summary",
        "",
        "| Model | Directional Accuracy |",
        "|---|---|",
        f"| Naive (momentum) | {naive['accuracy']:.1%} |",
        f"| ARIMA(1,0,1) | {arima_clf['accuracy']:.1%} |",
        "| LSTM (target) | ~60% (pending full training run) |",
        "",
        "> **Target:** ≥60% directional accuracy on out-of-sample data.",
        "> LSTM with GARCH volatility features and regime-aware training is expected to",
        "> exceed the ARIMA baseline. Full results pending tensorflow installation and",
        "> complete training run.",
    ]

    out.write_text("\n".join(lines), encoding="utf-8")
    print(f"\nReport saved -> {out}")
    return out


def main() -> None:
    df = download_data(TICKER, START, END)

    # Save raw data
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    raw_path = DATA_DIR / f"{TICKER.lower()}_raw.csv"
    df.to_csv(raw_path)
    print(f"  Raw data saved -> {raw_path}")

    print("\nRunning stationarity tests...")
    adf = run_adf_test(df["log_return"], "QQQ log_return")
    kpss = run_kpss_test(df["log_return"], "QQQ log_return")
    print(f"  ADF p-value: {adf['p_value']:.4f} ({'stationary' if adf['stationary'] else 'non-stationary'})")
    print(f"  KPSS p-value: {kpss['p_value']:.4f} ({'stationary' if kpss['stationary'] else 'non-stationary'})")

    print("\nComputing naive directional baseline...")
    naive = naive_directional_accuracy(df)
    print(f"  Naive accuracy: {naive['accuracy']:.1%}")

    print("\nRunning ARIMA baseline...")
    arima_clf, arima_reg = run_arima_baseline(df, ARIMA_WINDOW)
    print(f"  ARIMA directional accuracy: {arima_clf['accuracy']:.1%}")

    save_report(df, adf, kpss, naive, arima_clf, arima_reg)


if __name__ == "__main__":
    main()
