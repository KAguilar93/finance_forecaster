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
	python scripts/clean.py

clean-all:
	python scripts/clean.py --all

docker_build:
	docker build -f dockerfiles/train.dockerfile . -t train:v1
	docker build -f dockerfiles/predict.dockerfile -t predict:v1

docker_run:
	docker run --rm --name finance_forecaster_training trainer:v1
	docker run --rm --name finance_forecaster_predicting predicter:v1

docs:
	mkdocs serve
