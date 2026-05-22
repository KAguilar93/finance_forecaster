# Phase 2: Model Development

## Overview
This phase focuses on building, training, and validating machine learning models.

## Objectives

- [X] Implement baseline model
- [X] Train and evaluate initial models
- [X] Hyperparameter tuning
- [X] Cross-validation and performance analysis
- [X] Model comparison and selection

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

We have elected to implement our own proprietary model, made up of an ensemble as outlined below.

### Chosen Model
- Model Type: Ensemble model of ARIMA, GARCH, and LSTM training
- Best Hyperparameters:
- Performance Metrics: Greater than 55% Accuracy of trading recommendations

## Key Results

Our model pipeline is currently hitting ~57-58% accuracy of trading prediction recomendations.
We have included screenshots of running docker compose up from the command line with no local virtual
environment enabled (model runs entirely inside containers), as well as running our prediction pipeline with the make
commands, and running our model locally with a virtual environment enabled. Key to note is that regardless of the manner
the prediction pipeline was run the results were the same, pointing towards and supporting reproducibility.
(Which is why the screenshots all look the same but they were executed independently with different commands.)

![Results of running 'docker compose up --build' to force a new build](screenshots/docker%20compose%20up%20build%20results.png)
![Results of 'docker compose up' with prebuilt images](screenshots/docker%20compose%20up%20results.png)
![Results of running make docker_run to run full docker pipeline](screenshots/local%20make%20docker_run%20run%20with%20venv.png)
![Results of prediction pipeline using an activated virtual environment and make full command](screenshots/local%20make%20full%20run%20with%20venv.png)

## Challenges and Solutions

Getting our model to outperform an experienced trader, which is roughly ~55% trade prediction accuracy was difficult.
According to our research highly performant and experienced traders have a prediction accuracy of 55%, therefore
our prediction accuracy of ~57-58% outperforms great traders. Determining the correct ensemble and tuning of hyperparameters
for training was crucial to achieving our results. Our initial target of 60% accuracy was overly ambitious
and not rooted in reality, therefore we lowered our target to outperform great human traders, which we were able to
do, if only just.

Having our containers communicate with each other and persist the data required for training the model, then passing the
trained model for use in prediction was complex. Using the command line to mount and bind volumes and passing the location
of stored data and trained models was both cumbersome and error-prone. Therefore, we decided to create a docker-compose.yaml
file so that we could orchestrate the various containers, create a shared persistent volume where all data could be passed
between them, and a network to facilitate inter-container communication as a better solution. It also has the added benefit
of simplifying the commands someone outside of the development team is required to execute in order to run our forecasting
pipeline from end to end. It also allowed us to control the execution order of the containers, so that users of our pipeline
do not unintentionally run the prediction script before the model is trained, nor attempt to train the model before the required
data is pulled.


## Next Steps

Move to Phase 3 once model is selected and meets performance requirements.

## Status

- Start Date: 5-7-2026
- Estimated Completion: 5-20-2026
- Actual Completion: 5-21-2026
- Status: Completed

---

## How to Run & Test Phase 2

### Docker (Section 1)
```bash
# Build the image (Commands must be run from projet root folder where docker-compose.yaml and Makefile reside)
docker compose build  ->> [This builds all the individual containers needed for our finance forecast.]
or
make docker_build     ->> [Users can also use the make command which builds the requisite containers.]

# Run training with a volume mount so models persist
docker compose up     ->> [This runs the ensemble of orchestrated docker containers to run one after another, 
                            with a persitent volume and network attached to all of the containers so they can
                            communicate and share relevant data amongs each other. With the added benefit of the
                            full data, training, and prediction pipeline running entirely within containers and 
                            decoupled from local filesystems.]
or
make docker_run      ->> [Users can also use this make command to run the orchestrated docker containers]
```

## Section 2: Debugging Scenario

One significant issue encountered during development involved DVC authentication with the Google Drive remote used to store versioned datasets. Running `dvc pull` initially failed with the error `invalid_grant: Bad Request`, which indicated that a previously cached OAuth refresh token had expired or been revoked. The solution was to remove the stale credential file stored in the local PyDrive2 cache under the user's AppData directory and rerun `dvc pull`. This forced DVC to open a browser-based Google authentication flow and generate a new valid access token. After successful authentication, the raw and processed datasets were downloaded correctly from the shared Google Drive folder.

Additionally, we have included screenshots of using pdb and breakpoint() from the command line to step through and debug train_model.py.
As well as screenshots of using the vscode python debugger to view local variables in an effort to track why certain
modules were not being loaded correctly. There was an issue was due to running train_model.py locally without correctly
installing the requisite modules.

In order to run pdb fromt the command line on your desired python script you would like to step through and debug use:

```bash
python -m pdb [python_file.py]
```

The following are screenshots of various usages of pdb from command line to step through our python scripts.
Using breakpoint() in our scripts:
![pdb command line debugging with breakpoint in train_data 1.png](screenshots/pdb%20command%20line%20debugging%20with%20breakpoint%20in%20train_data%201.png)
![pdb command line debugging with breakpoint in train_data 2.png](screenshots/pdb%20command%20line%20debugging%20with%20breakpoint%20in%20train_data%202.png)
![pdb command line debugging with breakpoint in train_data 3.png](screenshots/pdb%20command%20line%20debugging%20with%20breakpoint%20in%20train_data%203.png)
![pdb command line debugging with breakpoint in train_data 4.png](screenshots/pdb%20command%20line%20debugging%20with%20breakpoint%20in%20train_data%204.png)

Without breakpoint() [To step through from beginning]:
![starting and stepping through and over make_data using pdb.png](screenshots/starting%20and%20stepping%20through%20and%20over%20make_data%20using%20pdb.png)
![starting and stepping through and over train_model using pdb 2.png](screenshots/starting%20and%20stepping%20through%20and%20over%20train_model%20using%20pdb%202.png)
![starting and stepping through and over train_model using pdb 3.png](screenshots/starting%20and%20stepping%20through%20and%20over%20train_model%20using%20pdb%203.png)

Using VSCode Python Debugger to catch an exception caused by incorrectly formed environment:
![python debugger in vscode.png](screenshots/python%20debugger%20in%20vscode.png)

## Section 3: Profiling Python and Machine Learning Code

Python's built-in `cProfile` module was used to analyze the performance of the training pipeline. The script `scripts/profile_training.py` executes `finance_forecaster.train_model.main()` under a profiler and saves both raw profiler output and a human-readable report to `reports/profiling/`.

### Profiling Commands

```bash
python scripts/profile_training.py
```

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

## Tool Orchestration
All of our tools work in concert in order to pull data from our data sources, train our model from the data, and make predictions
for next day financial movements of our identified index. We use a makefile in order to simplify command line executions and
remove humman error when running our prediction pipeline. Docker is used to containerize the entire prediction pipeline
to run entirely from within containers, with its own persistent volume in order to decouple our pipeline from local filesystems
and maintain reproducibility. Reproducibility is supported through the use of configuration management with Hydra, so that
we can easily configure hyperparameters for various experiments while fine tuning our model. MLFLow allows us to track the
various experiment configurations and experiments themselves so we can keep track of which model is best. We also use rich logging
to facilitate debugging and monitoring performance of our model over time. As well as pdb and vscode's python debugger
in order to trace potential bugs in our model, and to step through our python files as they execute line by line if needed
to track performance issues as shown by our profiler and monitoring configurations. Each tool plays a small part so that
altogether we can fine tune our model for the best results, track and manage our experiments so we can deploy the best model
in fully containerized pipeline anywhere we'd like, track the performance of our model over time, and have the required
logging tools at hand so we can be notified of performance degradation, and use the included debugging tools to identify
the root causes of any service interruptions or decreases in performance.
