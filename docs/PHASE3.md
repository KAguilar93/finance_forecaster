# Phase 3: Evaluation and Deployment

## Overview
This phase covers final evaluation, testing, and deployment preparation of the model.

## Objectives

- [ ] Final model evaluation on test set
- [ ] Production readiness assessment
- [ ] Documentation and knowledge transfer
- [ ] Deployment pipeline setup
- [ ] Monitoring and maintenance plan

## Deliverables

### 1. Final Evaluation Report
- Test set performance
- Model robustness analysis
- Edge case testing
- Performance summary

### 2. Deployment Artifacts
- Docker image created and tested
- Docker Compose configuration
- Container specifications documented
- API/inference server ready
- Configuration files documented

### 3. Documentation
- User guide for running predictions
- API documentation
- Deployment instructions
- Troubleshooting guide
- Model card

### 4. Monitoring and Maintenance
- Performance monitoring plan
- Model update strategy
- Data drift detection approach
- Feedback loop design

## Test Results

*To be filled in during Phase 3*

### Final Performance Metrics
- Test Accuracy:
- Test Loss:
- Other Metrics:

## Deployment Plan

Continuous Live Deployment on Push to Main branch
Smoke Test Deployments on Push to development-staging branch
(Development staging is used to merge all new feature branches before
live deployment to main for user service)

UI is live deployed to Hugging Face Spaces at: https://financeforecaster-financeforecasterapp.hf.space/?__theme=system&deep_link=zrF7v-q-54M

All container images required to run the app are pushed to Google Artifact Registry at: us-central1-docker.pkg.dev/finance-forecasters-pipeline/finance-forecaster-docker-containers
Cloud Function Endpoint can be found at: https://finance-forecaster-app-880001820336.us-central1.run.app/

Cloud Run is deploying on push and attempts to run the containers but container service start is broken 
therefore no endpoint of URL location can be given.

Follow these steps for setting up a GCS Artifact Registry and Pushing and Pulling container images to Artifact Registry:
(Note this is also done automatically on pushing updates to main branch)

### Pre-Requisites:

GCP CLI installed\
GCP Account Created with 1 GPU and the following services enabled:\
compute\
storage\
artifactregistry\
cloudbuild\
run\
cloudfunctions\
iam\
aiplatform\
GCP IAM roles assigned

To Create GCloud Artifact Registry run:

	gcloud artifacts repositories create finance-forecaster-docker-containers \
		--repository-format=docker \
		--location=us-central1 \
		--description="MLOps Finance Forecaster Docker Container registry"

To push/pull from this registry use the following URI:
us-central1-docker.pkg.dev/finance-forecasters-pipeline/finance-forecaster-docker-containers

![gcp create artifact registry.png](screenshots/gcp%20create%20artifact%20registry.png)

![gcp artifact registry.png](screenshots/gcp%20artifact%20registry.png)

#### Training Job

Custom Training Job to train model on Compute Engine can be run with the following command from the gcloud CLI:
~~~    
    gcloud ai custom-jobs create --region=us-central1 --display-name=finance-forecaster-training --config="<full path to config_cpu.yaml>"
~~~

Upon running this command you receive a job id, copy the id and run this command to track progress:
~~~
    gcloud ai custom-jobs stream-logs projects/880001820336/locations/us-central1/customJobs/<job id>
~~~
![gcp run and track custom training job.png](screenshots/gcp%20run%20and%20track%20custom%20training%20job.png)

#### GCP Cloud Functions
Cloud Function that Runs FastAPI endpoint can be found and accessed here:
https://finance-forecaster-app-880001820336.us-central1.run.app/

#### GCP Cloud Run Deployment
Is unfortunately currently broken. Upon push to main branch GCP cloud run successfully
builds and pushes container images with tags to Artifact Registry. Then pulls the images
from the registry and attempts to start and run the containers, but the containerized app
is not listening on the port injected by GCP cloud run, therefore it does not currently serve.

![broken cloudbuild run.png](screenshots/broken%20cloudbuild%20run.png)

### Deployment Environment
- Platform: Google Cloud Platform & Hugging Face Spaces
- Configuration: cloudbuild.yaml
- Expected Latency: 15 min
- Resource Requirements: N/A deployment downloads data to GCS Bucket, 
Training on Google Compute Engine with Custom Job, Prediction Serving on Google Cloud

## Known Limitations

Currently only tracks and predicts one index: QQQ

## Future Improvements

- [ ] Fix Continous Deployments to successfully start and run containers (app not listening to correct ports)
- [ ] Support multiple ticker predictions
- [ ] Daily re-training on newest data

## Handoff Checklist

- [ ] All code documented and commented
- [ ] Tests passing (100% coverage)
- [ ] Docker image tested
- [ ] Documentation complete
- [ ] Model versioning implemented
- [ ] Performance monitoring set up
- [ ] Deployment runbook created
- [ ] Team training completed

## Status

- Start Date: 5/29/2026
- Estimated Completion: 6/7/2026
- Actual Completion: TBD
- Status: Deploys on Push but deployments are broken
