.PHONY: data train backtest full regime-help clean

# Default target
all: full

# Data pipeline
data:
	python -m finance_forecaster.data.make_dataset

# Train models (ARIMA, GARCH, LSTM)
train:
	python -m models.train_model

# Regime-aware backtest + next day prediction
backtest:
	python -m tests.regime_aware_backtest

# Run everything in sequence (recommended)
full:
	@echo "Starting Full Pipeline..."
	@$(MAKE) data
	@$(MAKE) train
	@$(MAKE) backtest
	@echo ""
	@echo "================================================================"
	@echo "FULL PIPELINE COMPLETED SUCCESSFULLY"
	@echo "================================================================"
	@echo "Check these folders:"
	@echo "   reports/               → Predictions and metrics"
	@echo "   reports/figures/       → All generated charts"
	@echo "   data/processed/        → Final processed data"
	@echo "================================================================"

# Show regimes explanation
regime-help:
	@echo "Opening ensemble regimes explanation..."
	open reports/ensemble_regimes_explanation.txt

# Clean outputs
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
