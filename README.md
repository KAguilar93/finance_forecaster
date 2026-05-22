# Finance Forecasters
## Kevin Aguilar, Shang Andrews, James Russo, Joseph Hughes
### [SE489] ML Engineering for Production (MLOps)

# finance_forecaster

Finance Forecasting ML Model Pipeline -- Predicting Next Day Financial Movements

## Team Information

- **Project Lead:** Finance Forecasters (financeforecasters.mlops489@gmail.com)
- **Team Members:** Shang Andrews (sandre15@depaul.edu); James Russo (jrusso13@depaul.edu);Joseph Hughes (jhughe50@depaul.edu); Kevin Aguilar (kaguila3@depaul.edu)

## Project Overview

finance_forecaster is a machine learning project that implements a Finance Forecasting ML Model Pipeline that aims to predict Next Day Financial Movements by using time-series analysis. Additionally, we are building a pipeline around our model to enable continuous training with fresh data, continuous integration of model improvements, and continuous delivery of prediction services while maintaining traceable and reproducible experiments for validation and verification of our models performance. We are aiming for our model and pipeline to provide 55% accuracy of next day financial movements.\

See [Phase 1 Project Proposal](docs/PHASE1.md) for the full project proposal, deliverable documentation, in depth and alternative
installation instructions.\
See [Phase 2 Project Instructions](docs/PHASE2.md) for full instructions on running our containerized pipeline, profiling, logging,
debugging samples, challenges & solutions, and key results from our Finance Forecasting Pipeline.

**Key Objectives:**
- [ ] Provide Next Day Finance Movement Preditions with ~55% Accuracy
- [ ] Automated Continuous Training, Integration, and Delivery of Prediction Services in a Portable Environment
- [ ] Maintain Traceability and Reproducibility of Model Predictions and Experiments for Third-Party Validation

## Architecture Diagram

<img width="1782" height="878" alt="arch_diagram" src="https://github.com/user-attachments/assets/e9565b10-6844-476b-a359-a7f7583ab861" />


## Phase Deliverables

### Phase 1: Project Design & Model Development
- See [PHASE1.md](PHASE1.md) for detailed checklist

### Phase 2: Containerization & Monitoring
- See [PHASE2.md](PHASE2.md) for detailed checklist

### Phase 3: CI/CD & Deployment
- See [PHASE3.md](PHASE3.md) for detailed checklist

## Setup Instructions

### Prerequisites
- Python 3.11+ installed
- Git installed
- DVC Installed
- Hydra installed
- UV installed
- Docker installed

### Installation

**Option 1: Using uv (recommended - faster)**
```bash
pip install uv
uv pip install -r requirements.txt
```

**Option 2: Using pip**
```bash
pip install -U pip
pip install -r requirements.txt
```

### Development Setup

```bash
# Install package with dev tools
pip install -e ".[dev]"

# Set up pre-commit hooks
pre-commit install

# Run tests to verify setup
pytest tests/
```

### Running the Pipeline

```bash
#Run the entire pipeline inside containers (Both automatically build and run the containers with our pipeline)
make docker_run
or
docker compose up

# Prepare data (downloads QQQ + market features)
make data

# Train the model (ARIMA/LSTM + MLflow tracking)
make train

# Run regime-aware backtest + next-day prediction
make backtest

# Run full pipeline in one command
make full

# Generate predictions
make predict

# See all available commands
make help
```

Expected Output:
```
Loaded processed data: (2855, 37)

Running Ensemble (ARIMA + GARCH + LSTM) + Regime-Aware Backtest...


=== NEXT DAY ENSEMBLE PREDICTION ===

Current Regime       : low_vol_uptrend

Ensemble Prob UP     : 0.9000

Recommendation       : BUY - STRONG UP


=== TRADE PERFORMANCE ===

Total Trades Taken   : 1261

Hit Rate (Accuracy)  : 57.1%

TRADE ACCURACY IS 55% OR GREATER

Trade performance saved to: reports/trade_performance.txt

Next day prediction saved to: reports/next_day_ensemble_prediction.txt

Equity curve saved to reports/figures/ensemble_regime_backtest.png
Screenshots of running both commands to run entire containerized Forecasting Pipeline
```
Sample Commands for running pipeline [from project root folder where docker-compose.yaml and makefile reside]:

![docker compose up](docs/screenshots/docker%20compose%20up%20comand.png)
![make docker_run](docs/screenshots/make%20docker_run%20comand.png)


## Technology Stack

### Core Dependencies
- **numpy** == 2.4.4 - Numerical computing
- **pandas** == 2.3.3 - Data manipulation
- **scikit-learn** >= 1.8.0 - Machine learning algorithms
- **matplotlib** >= 3.10.9 - Visualization
- **tqdm** >= 4.67.3 - Progress bars
- **pyyaml** >= 6.0.3 - Configuration files

### Experiment Tracking
- **mlflow** >= 3.11.1 - MLflow experiment tracking

### Configuration Management
- **hydra-core** >= 1.3.2 - Hydra configuration framework
- **omegaconf** >= 2.3.0 - Hierarchical configuration

### Data Version Control
- **dvc** >= 3.67.1 - Data Version Control

### Financial Data Sources
- **yfinance** == 1.3.0

### Statistical/time-series modeling
- **statsmodels** == 0.14.6
- **arch** == 8.0.0

### API
- **fastapi** == 0.136.1
- **uvicorn** == 0.46.0

### Model persistence / utilities
- **joblib** ==1.5.3
- **python-dotenv** ==1.2.2

### Development Tools
- **pytest** >= 8.0 - Testing framework
- **pytest-cov** >= 5.0 - Code coverage
- **ruff** >= 0.6.0 - Linting and formatting
- **mypy** >= 1.11 - Static type checking
- **pre-commit** >= 3.8 - Git hooks framework

## Project Structure

This template uses the modern **`src/` layout** — the importable package lives in `src/finance_forecaster/`, decoupled from the repository root. That forces `pip install -e .` before imports work, which catches packaging bugs early.

```
finance_forecaster/                  # Repository root
├── src/
│   └── finance_forecaster/          # Importable Python package
│       ├── __init__.py                # Version + package metadata
│       ├── config.py                  # Paths & typed config (PROJECT_ROOT, TrainingConfig, ...)
│       ├── logging_config.py          # setup_logging() + get_logger()
│       ├── data/
│       │   ├── __init__.py
│       │   ├── loaders.py             # load_raw / load_processed / save_processed
│       │   └── make_dataset.py        # Raw → processed pipeline CLI
│       ├── features/
│       │   ├── __init__.py
│       │   └── build_features.py      # Feature engineering
│       ├── models/
│       │   ├── __init__.py
│       │   ├── base.py                # BaseModel ABC (fit/predict/save/load)
│       │   └── model.py               # Concrete Model scaffold
│       ├── evaluation/
│       │   ├── __init__.py
│       │   └── metrics.py             # classification_report, regression_report
│       ├── visualization/
│       │   ├── __init__.py
│       │   └── visualize.py           # Plot helpers
│       ├── utils/
│       │   ├── __init__.py
│       │   ├── io.py                  # JSON helpers
│       │   └── seed.py                # set_seed for reproducibility
│       ├── train_model.py             # Training CLI
│       └── predict_model.py           # Inference CLI
├── tests/                             # Unit and integration tests
│   ├── conftest.py
│   └── test_model.py
├── data/
│   ├── raw/                           # Immutable raw data
│   └── processed/                     # Cleaned, transformed data
├── models/                            # Trained model artifacts (.joblib)
├── notebooks/                         # Jupyter notebooks for exploration
├── reports/
│   └── figures/                       # Generated analysis and figures
├── docs/                              # MkDocs documentation
│   ├── mkdocs.yml
│   ├── index.md
│   ├── getting_started.md
│   └── api.md
├── dockerfiles/                       # Docker configuration
│   └── Dockerfile
├── configs/                           # Hydra configuration (if selected)
│   └── config.yaml
├── api/                               # FastAPI service (if selected)
├── .github/workflows/                 # GitHub Actions CI/CD
│   └── ci.yml
├── PHASE1.md                          # Phase 1 deliverables checklist
├── PHASE2.md                          # Phase 2 deliverables checklist
├── PHASE3.md                          # Phase 3 deliverables checklist
├── .pre-commit-config.yaml            # Pre-commit hooks (Ruff, mypy)
├── Makefile                           # Common commands
├── docker-compose.yaml                # Docker Compose setup
├── pyproject.toml                     # Project config & dependencies
├── requirements.txt                   # Runtime dependencies
├── LICENSE
└── README.md
```

### Why `src/` layout?

| | `src/` layout (this template) | Flat layout |
|---|---|---|
| Forces `pip install -e .` before import | ✅ | ❌ |
| Catches packaging bugs early | ✅ | ❌ |
| Adopted by | attrs, httpx, pydantic, flask, sqlalchemy | Older data-science templates |

Data and model artifacts are accessed via the constants in `finance_forecaster.config` (`PROJECT_ROOT`, `DATA_DIR`, `MODELS_DIR`, …) rather than relative paths — code is independent of where you invoke it from.

## Common Commands

```bash
# Install package + runtime dependencies (editable install)
make install

# Install dev tools + pre-commit hooks
make dev

# Run linting and formatting checks
make lint

# Auto-format code
make format

# Run tests
make test

# Clean up build artifacts
make clean

# Docker operations
make docker_build
make docker_run

# Serve documentation locally
make docs
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

## Contribution Summary
We are taking on a collaborative approach to this project, where each member will
touch and collaborate on each portion of the project. In order to better understand
the core concepts and mechanisms behind each portion of the model and pipeline.

While our entire team aided and assisted each other on the various aspects of the project it can
be said that:
Shang Andrews led: Data pipeline construction, profiling, configuration management
James Russo led: Data identification and cleaning, feature engineering, model development and fine tuning
Joseph Hughes led: Pipeline architecture, experiment tracking and logging, monitoring and debugging 
Kevin Aguilar led: Project structure and git branching strategies, containerization, and documentation

## References

- [Project Documentation](docs/index.md)
- [Phase 1 — Project Design & Model Development](PHASE1.md)
- [Phase 2 — Containerization & Monitoring](PHASE2.md)
- [Phase 3 — CI/CD & Deployment](PHASE3.md)

## License

This project is licensed under the MIT License. See [LICENSE](LICENSE) for details.
