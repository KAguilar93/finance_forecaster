# Getting Started with finance_forecaster

## Prerequisites

- Python 3.11 or higher
- pip or uv package manager
- Git (for version control)
- Docker (optional, for containerized execution)

## Installation

### Step 1: Clone the Repository

```bash
git clone <repository-url>
cd finance_forecaster
```

### Step 2: Create a Virtual Environment

**Using venv:**
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

**Using conda:**
```bash
conda create -n finance_forecaster python=3.11
conda activate finance_forecaster
```

### Step 3: Install Dependencies

**Using uv (recommended):**
```bash
pip install uv
uv pip install -r requirements.txt
```

**Using pip:**
```bash
pip install -U pip
pip install -r requirements.txt
```

### Step 4: Set Up Development Environment

```bash
# Install package in editable mode (includes dev tools)
pip install -e ".[dev]"

# Install pre-commit hooks
pre-commit install

# Run tests to verify setup
pytest tests/
```

## Running the Project

### Data Download

Download QQQ and market feature data from yfinance (tries DVC/Google Drive cache first):

```bash
make data
```

This saves raw data to `data/raw/` and processed data to `data/processed/`.

### Model Training

Train an ARIMA or LSTM model and log the run to MLflow:

```bash
make train
# or with custom parameters:
python -m finance_forecaster.train_model --arima-p 1 --arima-d 0 --arima-q 1
```

### Backtest

Run the regime-aware ensemble backtest and generate prediction outputs:

```bash
make backtest
```

Outputs: `reports/next_day_ensemble_prediction.txt`, `reports/trade_performance.txt`, `reports/figures/`

### Run Full Pipeline

Run data → train → backtest in one command:

```bash
make full
```

### Model Prediction

Generate next-day directional predictions:

```bash
python -m finance_forecaster.predict_model --input data/raw/qqq_raw.csv --output predictions/predictions.csv
```

## Development Workflow

### Running Tests

```bash
# Run all tests
make test

# Run tests with coverage
pytest tests/ --cov=finance_forecaster

# Run specific test file
pytest tests/test_model.py -v
```

### Code Quality

```bash
# Check for linting issues
make lint

# Auto-format and fix issues
make format

# Type checking
mypy src
```

### Pre-commit Hooks

Pre-commit hooks automatically run before each commit:

```bash
# Manually run pre-commit checks
pre-commit run --all-files

# Update hooks
pre-commit autoupdate
```

## Docker Usage

### Build Docker Image

```bash
make docker_build
```

Or manually:

```bash
docker build -t finance_forecaster -f dockerfiles/Dockerfile .
```

### Run with Docker Compose

```bash
docker-compose up
```

### Run Single Container

```bash
make docker_run
```

## Project Structure

```
finance_forecaster/                  # Repository root
├── src/
│   └── finance_forecaster/          # Importable package (src/ layout)
│       ├── config.py                  # Paths & typed config
│       ├── logging_config.py
│       ├── data/                      # Loaders + raw→processed pipeline
│       ├── features/                  # Feature engineering
│       ├── models/                    # BaseModel + concrete Model
│       ├── evaluation/                # Metrics
│       ├── visualization/
│       ├── utils/                     # seed, io
│       ├── train_model.py
│       └── predict_model.py
├── tests/                             # Unit tests
├── data/                              # raw/ and processed/
├── models/                            # Trained model artifacts
├── docs/                              # MkDocs documentation
├── configs/                           # Hydra configuration (optional)
├── pyproject.toml
├── requirements.txt
└── Makefile
```

## Configuration

### Using Hydra Configuration
Configuration defaults live in `configs/config.yaml` and cover model, data, training, logging, and MLflow settings. The training CLI uses argparse for direct overrides:

```bash
# Train with a different ARIMA order
python -m finance_forecaster.train_model --arima-p 2 --arima-d 0 --arima-q 2

# Change the train/test split
python -m finance_forecaster.train_model --test-split 0.15
```

## Cleaning Generated Files

Remove logs, MLflow runs, predictions, and cache files:

```bash
python scripts/clean.py
# or
make clean
```

Full clean — also removes `data/processed/`, `models/`, and `reports/` outputs:

```bash
python scripts/clean.py --all
# or
make clean-all
```

## Troubleshooting

### Import Errors

If you get `ModuleNotFoundError`, ensure:
1. Virtual environment is activated
2. Dependencies are installed: `pip install -r requirements.txt`
3. Package is installed in editable mode: `pip install -e .`

### Pre-commit Hook Failures

If pre-commit hooks fail:

```bash
# See what failed
pre-commit run --all-files

# Fix issues manually or with auto-fix
make format

# Try committing again
```

## Next Steps

1. Review the [documentation](index.md)
2. Start with [Phase 1](PHASE1.md) - Data Exploration
3. Check the [API Reference](api.md)

## Support

For issues and questions:
- Check existing [documentation](index.md)
- Review [Phase deliverables](PHASE1.md)
- Contact Finance Forecasters (financeforecasters.mlops489@gmail.com)
