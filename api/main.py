"""Finance Forecaster FastAPI service.

Serves next-day QQQ movement predictions from pre-generated report files
produced by `make full`.  No live inference at request time keeps latency low
and avoids dependency on the full model stack at serve time.
"""

from __future__ import annotations

import os
from pathlib import Path

from fastapi import FastAPI, HTTPException

from api.schemas import PredictRequest, PredictResponse

app = FastAPI(
    title="Finance Forecaster API",
    description="Next-day QQQ movement prediction service",
    version="1.0.0",
)

_REPORT_PATH = Path(os.environ.get("PREDICTION_REPORT_PATH", "reports/next_day_ensemble_prediction.txt"))


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/predict", response_model=PredictResponse)
def predict(request: PredictRequest) -> PredictResponse:
    if not _REPORT_PATH.exists():
        raise HTTPException(
            status_code=503,
            detail="Model not yet trained. Run `make full` first.",
        )

    prob_up = 0.0
    recommendation = "UNKNOWN"

    for line in _REPORT_PATH.read_text().splitlines():
        if "Ensemble Prob UP" in line:
            try:
                prob_up = float(line.split(":")[-1].strip())
            except ValueError:
                pass
        if "Recommendation" in line:
            recommendation = line.split(":")[-1].strip()

    return PredictResponse(
        ticker=request.ticker,
        prediction="UP" if prob_up >= 0.5 else "DOWN",
        probability_up=prob_up,
        recommendation=recommendation,
    )
