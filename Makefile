.PHONY: install dev data train predict backtest full test lint format clean docker_build docker_run docs regime-help

# Note: 'uv' is a faster alternative to pip. Install with: pip install uv
# Then replace 'pip install' with 'uv pip install' in the commands below.

install:
	pip install -U pip
	pip install -r requirements.txt
	pip install -e .

dev: install
	pip install -r requirements_dev.txt
	pre-commit install

data:
	python -m finance_forecaster.data.make_dataset

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
	docker build -t finance_forecaster -f dockerfiles/Dockerfile .

docker_run:
	docker run --rm finance_forecaster

docs:
	mkdocs serve
