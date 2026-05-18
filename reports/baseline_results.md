# Baseline Model Results

**Run date:** 2026-05-16 17:55
**Ticker:** QQQ
**Period:** 2015-01-01 -> 2024-12-31
**Trading days:** 2514

---

## 1. Data Summary

| Stat | Value |
|---|---|
| Mean daily return | 0.000672 |
| Std daily return | 0.013783 |
| Min daily return | -0.127592 |
| Max daily return | 0.081309 |
| % Up days | 56.0% |

---

## 2. Stationarity Tests (Log Returns)

| Test | Statistic | p-value | Stationary? |
|---|---|---|---|
| ADF | -16.2463 | 0.0000 | Yes |
| KPSS | 0.0460 | 0.1000 | Yes |

> Both tests confirm log returns are stationary — appropriate for ARIMA modeling.

---

## 3. Naive Directional Baseline

> Predict tomorrow's direction = today's direction (momentum baseline).

| Metric | Value |
|---|---|
| Accuracy | 0.500 |
| Precision | 0.500 |
| Recall | 0.500 |
| F1 | 0.500 |

---

## 4. ARIMA(1,0,1) Rolling Forecast Baseline

> Evaluated on last 60 trading days with rolling re-fit.

### Directional Accuracy

| Metric | Value |
|---|---|
| Accuracy | 0.550 |
| Precision | 0.537 |
| Recall | 0.550 |
| F1 | 0.540 |

### Return Magnitude (Regression)

| Metric | Value |
|---|---|
| MAE | 0.9040 |
| RMSE | 1.2053 |
| R² | -0.2171 |

---

## 5. Summary

| Model | Directional Accuracy |
|---|---|
| Naive (momentum) | 50.0% |
| ARIMA(1,0,1) | 55.0% |
| LSTM (target) | ~60% (pending full training run) |

> **Target:** ≥60% directional accuracy on out-of-sample data.
> LSTM with GARCH volatility features and regime-aware training is expected to
> exceed the ARIMA baseline. Full results pending tensorflow installation and
> complete training run.
