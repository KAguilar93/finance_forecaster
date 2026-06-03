"""Tests for the Finance Forecaster FastAPI service."""

from __future__ import annotations

from pathlib import Path

import pytest
from fastapi.testclient import TestClient

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_report(tmp_path: Path, prob: float = 0.65, rec: str = "BUY") -> Path:
    """Write a minimal prediction report file and return its path."""
    report = tmp_path / "next_day_ensemble_prediction.txt"
    report.write_text(f"Ensemble Prob UP: {prob}\nRecommendation: {rec}\n")
    return report


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def client_with_report(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> TestClient:
    """TestClient with a pre-generated prediction report available."""
    report = _make_report(tmp_path)
    monkeypatch.setenv("PREDICTION_REPORT_PATH", str(report))

    # Re-import app after env var is set so the module-level path is refreshed.
    import importlib

    import api.main as main_mod

    importlib.reload(main_mod)
    return TestClient(main_mod.app)


@pytest.fixture
def client_no_report(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> TestClient:
    """TestClient where no prediction report exists."""
    monkeypatch.setenv("PREDICTION_REPORT_PATH", str(tmp_path / "missing.txt"))

    import importlib

    import api.main as main_mod

    importlib.reload(main_mod)
    return TestClient(main_mod.app)


# ---------------------------------------------------------------------------
# /health
# ---------------------------------------------------------------------------


def test_health(client_with_report: TestClient) -> None:
    response = client_with_report.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


# ---------------------------------------------------------------------------
# /predict
# ---------------------------------------------------------------------------


def test_predict_returns_200(client_with_report: TestClient) -> None:
    response = client_with_report.post("/predict", json={"ticker": "QQQ"})
    assert response.status_code == 200


def test_predict_response_has_required_keys(client_with_report: TestClient) -> None:
    data = client_with_report.post("/predict", json={"ticker": "QQQ"}).json()
    assert {"ticker", "prediction", "probability_up", "recommendation"} <= data.keys()


def test_predict_up_when_prob_above_half(client_with_report: TestClient) -> None:
    data = client_with_report.post("/predict", json={"ticker": "QQQ"}).json()
    assert data["prediction"] == "UP"
    assert data["probability_up"] == pytest.approx(0.65)


def test_predict_down_when_prob_below_half(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    report = _make_report(tmp_path, prob=0.40, rec="SELL")
    monkeypatch.setenv("PREDICTION_REPORT_PATH", str(report))

    import importlib

    import api.main as main_mod

    importlib.reload(main_mod)
    client = TestClient(main_mod.app)
    data = client.post("/predict", json={"ticker": "QQQ"}).json()
    assert data["prediction"] == "DOWN"


def test_predict_503_when_no_report(client_no_report: TestClient) -> None:
    response = client_no_report.post("/predict", json={"ticker": "QQQ"})
    assert response.status_code == 503


def test_predict_ticker_echoed(client_with_report: TestClient) -> None:
    data = client_with_report.post("/predict", json={"ticker": "SPY"}).json()
    assert data["ticker"] == "SPY"
