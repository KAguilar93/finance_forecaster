# Phase 2: Model Development

## Overview
This phase focuses on building, training, and validating machine learning models.

## Objectives

- [ ] Implement baseline model
- [ ] Train and evaluate initial models
- [ ] Hyperparameter tuning
- [ ] Cross-validation and performance analysis
- [ ] Model comparison and selection

## Deliverables

### 1. Model Implementation
- Model architecture defined
- Training pipeline implemented
- Evaluation metrics chosen
- Baseline performance established

### 2. Experiment Tracking
- All experiments logged and documented
- MLflow experiment tracking configured

### 3. Performance Analysis
- Model comparison results
- Hyperparameter sensitivity analysis
- Feature importance analysis
- Error analysis and patterns

### 4. Model Artifacts
- Best model saved and versioned
- Model evaluation report
- Training curves and visualizations
- Configuration documentation

## Model Selection

*To be filled in during Phase 2*

### Chosen Model
- Model Type:
- Best Hyperparameters:
- Performance Metrics:

## Key Results

*To be filled in during Phase 2*

## Challenges and Solutions

*To be filled in during Phase 2*

## Next Steps

Move to Phase 3 once model is selected and meets performance requirements.

## Status

- Start Date:
- Estimated Completion:
- Actual Completion:
- Status: Not Started

---

## How to Run & Test Phase 2

### Docker (Section 1)
```bash
# Build the image
docker build -t finance-forecaster -f dockerfiles/Dockerfile .

# Run training with a volume mount so models persist
docker run --rm -v ${PWD}/models:/app/models finance-forecaster
```

## Section 2: Debugging Scenario

One significant issue encountered during development involved DVC authentication with the Google Drive remote used to store versioned datasets. Running `dvc pull` initially failed with the error `invalid_grant: Bad Request`, which indicated that a previously cached OAuth refresh token had expired or been revoked. The solution was to remove the stale credential file stored in the local PyDrive2 cache under the user's AppData directory and rerun `dvc pull`. This forced DVC to open a browser-based Google authentication flow and generate a new valid access token. After successful authentication, the raw and processed datasets were downloaded correctly from the shared Google Drive folder.

## Section 3: Profiling Python and Machine Learning Code

Python's built-in `cProfile` module was used to analyze the performance of the training pipeline. The script `scripts/profile_training.py` executes `finance_forecaster.train_model.main()` under a profiler and saves both raw profiler output and a human-readable report to `reports/profiling/`.

### Profiling Commands

```bash
python scripts/profile_training.py

### MLflow Experiment Tracking (Section 4)
```bash
# Run a training experiment (ARIMA(1,0,1))
python -m finance_forecaster.train_model --arima-p 1 --arima-d 0 --arima-q 1

# Run a second experiment to compare (ARIMA(2,0,2))
python -m finance_forecaster.train_model --arima-p 2 --arima-d 0 --arima-q 2

# Open the MLflow UI to compare runs
mlflow ui
# Navigate to http://127.0.0.1:5000
```

### Logging (Section 5)
Logging is configured automatically when any entrypoint runs. Logs are written to both the console (colored via rich) and `logs/app.log` (rotating, 10 MB × 5 backups):
```bash
python -m finance_forecaster.predict_model --input data/raw/qqq_raw.csv --output predictions/predictions.csv
# Colored log output appears in terminal
# Structured log written to logs/app.log
```

### Run All Tests
```bash
pytest tests/ -v
```
