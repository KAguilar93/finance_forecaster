"""Tests for data pipeline utilities."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from finance_forecaster.evaluation.metrics import classification_report, regression_report
from finance_forecaster.features.build_features import build_features
from finance_forecaster.utils.io import load_json, save_json

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def ohlcv_df() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "Open": [400.0, 401.0, 402.0],
            "High": [405.0, 406.0, 407.0],
            "Low": [398.0, 399.0, 400.0],
            "Close": [403.0, 404.0, 405.0],
            "Volume": [1_000_000, 1_100_000, 1_200_000],
        }
    )


# ---------------------------------------------------------------------------
# build_features
# ---------------------------------------------------------------------------


def test_build_features_returns_dataframe(ohlcv_df: pd.DataFrame) -> None:
    result = build_features(ohlcv_df)
    assert isinstance(result, pd.DataFrame)


def test_build_features_preserves_rows(ohlcv_df: pd.DataFrame) -> None:
    result = build_features(ohlcv_df)
    assert len(result) == len(ohlcv_df)


def test_build_features_does_not_mutate_input(ohlcv_df: pd.DataFrame) -> None:
    original_cols = list(ohlcv_df.columns)
    build_features(ohlcv_df)
    assert list(ohlcv_df.columns) == original_cols


# ---------------------------------------------------------------------------
# classification_report
# ---------------------------------------------------------------------------


def test_classification_report_keys() -> None:
    y = [1, 0, 1, 0]
    report = classification_report(y, y)
    assert set(report.keys()) == {"accuracy", "precision", "recall", "f1"}


def test_classification_report_perfect_accuracy() -> None:
    y = [1, 0, 1, 1, 0]
    report = classification_report(y, y)
    assert report["accuracy"] == pytest.approx(1.0)


def test_classification_report_returns_floats() -> None:
    y_true = [1, 0, 1, 0, 1]
    y_pred = [1, 0, 0, 0, 1]
    report = classification_report(y_true, y_pred)
    for v in report.values():
        assert isinstance(v, float)


# ---------------------------------------------------------------------------
# regression_report
# ---------------------------------------------------------------------------


def test_regression_report_keys() -> None:
    y = [1.0, 2.0, 3.0]
    report = regression_report(y, y)
    assert set(report.keys()) == {"mae", "mse", "rmse", "r2"}


def test_regression_report_perfect_fit() -> None:
    y = [1.0, 2.0, 3.0]
    report = regression_report(y, y)
    assert report["mae"] == pytest.approx(0.0)
    assert report["r2"] == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# save_json / load_json round-trip
# ---------------------------------------------------------------------------


def test_json_roundtrip(tmp_path: Path) -> None:
    data = {"accuracy": 0.87, "labels": [0, 1]}
    path = tmp_path / "metrics.json"
    save_json(data, path)
    loaded = load_json(path)
    assert loaded == data


def test_save_json_creates_parent_dirs(tmp_path: Path) -> None:
    path = tmp_path / "nested" / "dir" / "out.json"
    save_json({"x": 1}, path)
    assert path.exists()
