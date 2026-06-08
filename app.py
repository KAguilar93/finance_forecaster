"""Gradio UI for the Finance Forecaster prediction service.

Calls the Cloud Run backend (or a local FastAPI server) and displays
the next-day QQQ movement prediction in the browser.
"""

from __future__ import annotations

import os

import gradio as gr
import requests

CLOUD_RUN_URL = os.environ.get(
    "CLOUD_RUN_URL",
    "https://finance-forecaster-api-xxxx-uc.a.run.app",
)


def get_prediction(ticker: str) -> str:
    try:
        response = requests.post(
            f"{CLOUD_RUN_URL}/predict",
            json={"ticker": ticker},
            timeout=30,
        )
        response.raise_for_status()
        data = response.json()
        return (
            f"**Ticker:** {data['ticker']}\n\n"
            f"**Prediction:** {data['prediction']}\n\n"
            f"**Probability UP:** {data['probability_up']:.1%}\n\n"
            f"**Recommendation:** {data['recommendation']}"
        )
    except Exception as e:
        return f"Error calling prediction service: {e}"


demo = gr.Interface(
    fn=get_prediction,
    inputs=gr.Textbox(value="QQQ", label="Ticker Symbol"),
    outputs=gr.Markdown(label="Prediction"),
    title="Finance Forecaster",
    description=(
        "Next-day QQQ movement prediction using ARIMA + GARCH + LSTM ensemble "
        "with regime-aware backtesting. Powered by the Finance Forecasters MLOps "
        "pipeline (SE 489, DePaul University)."
    ),
    examples=[["QQQ"]],
)

if __name__ == "__main__":
    demo.launch()
