# PHASE 1: Project Design & Model Development

**Team:** Kevin Aguilar, Shang Andrews, James Russo, Joseph Hughes
**Course:** SE489 — ML Engineering for Production (MLOps)

---

## Project Proposal

### Problem Statement

Finance Forecaster aims to aid financial decisions in stock market trading by predicting next-day financial movements on the market index or stock of a user's choice. Enabling traders to make informed decisions on their purchasing or selling actions based on historical market data, as analyzed and interpreted by a machine learning pipeline. Through our model pipeline we seek to help financial professionals and enthusiasts make informed business decisions on their investments.

Making decisions on whether to buy or sell puts, calls, individual stock or index shares can feel like a coin flip; Finance Forecasters seeks to help decision makers make confident decisions by providing predictions on whether their index or stock of choice will move up or down the following day. Currently financial professionals utilize hard-earned experience and manual review of data to make their decisions — a process that is error-prone due to human limitations and the scale of manual data analysis. By using time-series analysis along with historical multi-year datasets pulled from well-known and vetted sources, Finance Forecasters will provide next-day predictions on individual share movements, making educated market decisions accessible to professionals of all experience levels.

### Project Objectives and Expected Impact

Finance Forecaster leverages Machine Learning to analyze historical market data for next-day price directionality, functioning as a short-term predictive agent delivering trade recommendations based on predicted confidence of a stock's price and directional movement.

The model is initially built around the Invesco QQQ Trust index fund (an ETF tracking the Nasdaq 100), which provides a robust development and benchmarking baseline. The model architecture is designed to allow dynamic selection of any publicly traded company accessible via the `yfinance` library.

The methodology employs a multifaceted approach evaluating ARIMA and GARCH for statistical and volatility forecasting in conjunction with LSTM for non-linear pattern recognition. These individual estimators may be integrated into a hybrid ensemble framework to maximize predictive accuracy and resilience.

### Success Metrics

Performance is measured through classification and regression reports. The minimum target is **60% directional accuracy** matching model output to actual market movements. Results will also be compared against existing financial model baselines.

### Dataset

- **Source:** [yfinance](https://pypi.org/project/yfinance/) — publicly available, freely accessible, and historically thorough
- **Primary instrument:** QQQ (Invesco QQQ Trust, Nasdaq 100 ETF)
- **Access:** `data/download.py` fetches OHLCV data in the required format
- **Scope:** Multi-year daily OHLCV data; model architecture supports dynamic ticker input

### Model Considerations

| Model | Role |
|---|---|
| ARIMA | Statistical time-series baseline; captures linear autocorrelation |
| GARCH | Volatility forecasting; expanding-window regime detection |
| LSTM | Non-linear sequence modeling; captures long-range dependencies |

Models are implemented in `src/finance_forecaster/models/` with a shared `BaseModel` interface for fit/predict/save/load.

### Open-Source Tools

| Tool | Purpose |
|---|---|
| `yfinance` | Market data ingestion |
| `statsmodels` | ARIMA modeling and stationarity tests (ADF, KPSS) |
| `arch` | GARCH volatility modeling |
| `tensorflow` / `keras` | LSTM sequence modeling |
| `scikit-learn` | Feature preprocessing, evaluation metrics |
| `mlflow` | Experiment tracking |
| `hydra-core` | Configuration management |
| `dvc` | Data version control |
| `fastapi` / `uvicorn` | Prediction API (Phase 3) |

---

## 1. Project Proposal

- [x] **Scope & Objectives**: Next-day directional movement prediction with 55% accuracy target
- [x] **Detailed Description**: 300+ word proposal above covering business context, technical approach, and expected outcomes
- [x] **Dataset Selection**: yfinance / QQQ documented with justification
- [x] **Dataset Description**: OHLCV daily data, multi-year history, dynamic ticker support
- [x] **Model Considerations**: ARIMA, GARCH, LSTM evaluated and documented
- [x] **Open-Source Tools**: All tools documented with purpose in table above

---

## 2. Code Organization & Setup

- [x] **GitHub Repository**: Cookiecutter MLOps structure with `src/`, `tests/`, `data/`, `models/`, `configs/`, `docs/`
- [x] **Environment Setup**: Python 3.11+ venv; documented in README
- [x] **Dependency Management**: `requirements.txt` and `pyproject.toml` maintained
- [x] **Project Structure**: Clear separation — `src/finance_forecaster/{data,models,features,evaluation,utils,visualization}/`
- [ ] **Version Pinning**: Core deps pinned in `pyproject.toml`; `requirements.txt` uses loose pins
- [x] **Installation Documentation**: `make install` / `make dev` documented in README

---

## 3. Version Control & Collaboration

- [x] **Regular Commits**: Descriptive commits across team branches
- [x] **Branching Strategy**: Feature branching — `development-staging`, `containerization`, `joe-dev`, `documentation`, etc.
- [x] **Pull Request Process**: PRs used for all merges to `main`
- [x] **Team Roles**: Four members defined; Finance Forecasters team email for coordination
- [ ] **Code Review Guidelines**: Not formally documented
- [x] **Commit History**: Clean, readable history maintained on `main`

---

## 4. Data Handling

- [x] **Data Cleaning Scripts**: `data/download.py` for ingestion; `src/finance_forecaster/data/make_dataset.py` for preprocessing
- [x] **Normalization**: Feature scaling implemented via StandardScaler in the pipeline
- [ ] **Data Augmentation**: Not applicable for financial time-series
- [x] **Data Documentation**: Dataset described in proposal section above
- [x] **Data Splits**: Train/validation/test split defined in `configs/config.yaml` (80/10/10)
- [ ] **Data Validation**: Not yet implemented as a standalone script
- [x] **DVC Setup**: DVC initialized; `data/raw/` and `data/processed/` tracked

---

## 5. Model Training

- [x] **Training Environment**: Local CPU/GPU environment; documented in README
- [x] **Baseline Model**: LSTM, GARCH, ARIMA implemented in `src/finance_forecaster/models/`
- [x] **Hyperparameter Configuration**: Baseline hyperparameters in `configs/config.yaml`
- [x] **Evaluation Metrics**: MSE, RMSE, MAE, directional accuracy in `src/finance_forecaster/evaluation/metrics.py`
- [x] **Model Persistence**: `model.save()` / `model.load()` interface via `BaseModel`; joblib serialization
- [x] **Training Reproducibility**: `src/finance_forecaster/utils/seed.py` — `set_seed()` called at entry point
- [x] **Performance Baseline**: See `reports/baseline_results.md` — ARIMA(1,0,1) achieves 55% directional accuracy vs 50% naive baseline

---

## 6. Documentation & Reporting

- [x] **README**: Comprehensive README with overview, setup, quick start, dependencies, contributing, license
- [x] **Code Docstrings**: Docstrings on all public functions and classes
- [x] **Code Style**: `ruff` configured in `pyproject.toml` with E/F/I/N/W/B/UP rules
- [x] **Type Hints**: Type hints throughout `src/finance_forecaster/`
- [x] **Type Checking**: `mypy` configured in `pyproject.toml`; passes on pre-commit
- [x] **Makefile**: `make install`, `make dev`, `make data`, `make train`, `make backtest`, `make full`, `make predict`, `make test`, `make lint`, `make format`, `make clean`, `make docker_build`, `make docker_run`, `make docs`
- [ ] **CONTRIBUTING.md**: Not yet created
- [ ] **API Documentation**: FastAPI docs auto-generated; MkDocs setup in progress

---

## Running Phase 1

### Prerequisites

- Python 3.11+
- Git
- Virtual environment activated (see [Getting Started](getting_started.md))

### Step 1: Install Dependencies

```bash
pip install -e ".[dev]"
```

### Step 2: Download Data

Download QQQ and market feature data (tries DVC/Google Drive cache first):

```bash
make data
```

This saves raw data to `data/raw/` and processed data with market features to `data/processed/`.

### Step 3: Train the Model

Fit an ARIMA model and log the run to MLflow:

```bash
python -m finance_forecaster.train_model --arima-p 1 --arima-d 0 --arima-q 1
```

### Step 4: Generate Predictions

Run next-day directional forecasting:

```bash
python -m finance_forecaster.predict_model --input data/raw/qqq_raw.csv --output predictions/predictions.csv
```

Expected output:
```
INFO  Loaded 2513 trading days
INFO  Running ARIMA(1,0,1) rolling forecast over 30 days...
INFO  Next-day forecast: 0.1863% -> UP
INFO  Rolling window accuracy (30 days): 53.3%
INFO  Next trading day prediction: UP (0.1863%)
```

### Step 5: Run Tests

```bash
pytest tests/ -v
```

---

## Development Workflow

### Code Quality

```bash
# Lint and format
make lint
make format

# Type checking
mypy src
```

### Pre-commit Hooks

Pre-commit hooks run automatically on each commit. To run manually:

```bash
pre-commit run --all-files
```

---

## Troubleshooting

### ModuleNotFoundError

Ensure the package is installed in editable mode:
```bash
pip install -e .
```

### No data file found

Run the data pipeline before training or predicting:
```bash
make data
```

### Pre-commit hook failures

Hooks may auto-fix files on first run — just re-stage and commit again:
```bash
git add -A
git commit -m "your message"
```

---

> See [README.md](../README.md) for full setup and usage instructions.
> See [reports/baseline_results.md](../reports/baseline_results.md) for baseline model evaluation results.
