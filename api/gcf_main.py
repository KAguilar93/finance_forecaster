"""Cloud Functions entry point — wraps the FastAPI /predict endpoint."""

from __future__ import annotations

import functions_framework

from api.main import PredictRequest, predict


@functions_framework.http
def predict_handler(request):  # type: ignore[no-untyped-def]
    body = request.get_json(silent=True) or {}
    req = PredictRequest(**body)
    result = predict(req)
    return result.model_dump(), 200, {"Content-Type": "application/json"}
