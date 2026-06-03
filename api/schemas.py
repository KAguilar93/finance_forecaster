"""Pydantic request/response models for the Finance Forecaster API."""

from __future__ import annotations

from pydantic import BaseModel


class PredictRequest(BaseModel):
    ticker: str = "QQQ"


class PredictResponse(BaseModel):
    ticker: str
    prediction: str
    probability_up: float
    recommendation: str
