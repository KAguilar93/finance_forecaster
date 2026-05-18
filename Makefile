.PHONY: install dev data train predict backtest full test lint format clean docker_build docker_run docs regime-help

# Default target
all: full

# Data pipeline
data:
	python -m finance_forecaster.data.make_dataset

# Train models (ARIMA, GARCH, LSTM)
train:
	python -m finance_forecaster.train_model

predict:
	python -m finance_forecaster.predict_model

backtest:
	python -m tests.regime_aware_backtest

full:
	@echo "Starting full pipeline..."
	@$(MAKE) data
	@$(MAKE) train
	@$(MAKE) backtest
	@echo "Pipeline complete. Check reports/ for outputs."

regime-help:
	python -c "print(open('reports/ensemble_regimes_explanation.txt').read())"

test:
	pytest tests/

lint:
	ruff check .
	ruff format --check .

format:
	ruff check --fix .
	ruff format .

clean:
	rm -rf data/processed/*.csv reports/*.csv reports/figures/*.png reports/*.txt models/*.pkl

help:
	@echo "Available commands:"
	@echo "  make data          → Download and process data"
	@echo "  make train         → Train ARIMA, GARCH, LSTM"
	@echo "  make backtest      → Run regime-aware backtest + next day prediction"
	@echo "  make full          → Run everything in order (recommended)"
	@echo "  make regime-help   → Show regime explanation"
	@echo "  make clean         → Remove generated files"
