# Fraud Detection MLOps Pipeline on AWS

## Overview
This project implements a production-style MLOps pipeline for fraud detection on AWS. It simulates a real-world system where models must continuously adapt to changing data distributions (concept drift). The pipeline automatically detects performance degradation, retrains models, and conditionally deploys improved models using a champion/challenger framework. The system integrates AWS Step Functions, Lambda, SageMaker, and S3 to orchestrate a fully automated, state-aware machine learning lifecycle.


## Problem Statement
Fraud detection systems must adapt to changing transaction behavior over time. This pipeline addresses that challenge by orchestrating automated model retraining and deployment decisions based on evaluation metrics and drift-aware workflow logic.

## Architecture
docs/screenshots/fraud_architecture.png


## Cloud Architecture Mapping

This project mirrors a production-style MLOps system using AWS:

### Data Layer (Amazon S3)
- raw/ → incoming datasets
- processed/ → feature-engineered data
- test/ → persistent combined test set for drift detection

### Model Layer (S3 + SageMaker)
- staging/ → model artifacts, evaluation results, preprocessing objects
- model.tar.gz → deployed model package

### Compute Layer
- AWS Step Functions → orchestrates workflow
- AWS Lambda → lightweight logic (drift + evaluation)
- SageMaker Processing → preprocessing + drift checks
- SageMaker Training → model training
- SageMaker Endpoint → real-time inference

### Observability
- CloudWatch logs used to debug pipeline across distributed services

## Workflow
1. Raw transaction data is stored in S3
2. Step Functions orchestrates the ML workflow
3. Preprocessing prepares training-ready features
4. Drift detection checks for concept/data changes
5. SageMaker trains a challenger model
6. The challenger is evaluated against the champion
7. If performance improves, the new model is deployed

## Tech Stack
- Python
- AWS Lambda
- AWS Step Functions
- Amazon SageMaker
- Amazon S3
- XGBoost
- CloudWatch

## Repository Structure

- `data/` – Raw and processed datasets
- `lambda/` – Lambda functions (drift detection, F1 evaluation)
- `sagemaker/` – Training, preprocessing, evaluation scripts
- `step_functions/` – State machine definitions
- `artifacts/` – Model artifacts and outputs
- `dashboard/` – (future) monitoring tools
- `tests/` – Unit/integration tests

## Results
- Automated retraining pipeline
- Model evaluation artifacts
- Champion/challenger selection
- Successful endpoint deployment

## Results

| Stage | F1 Score |
|------|--------|
| Initial Model (Base Dataset) | 0.79 |
| Stable Data (No Drift) | 0.76 |
| Concept Drift Dataset | 0.12 |
| Retrained Model | 0.21 |

**Outcome:**
- Successfully detected concept drift
- Triggered retraining pipeline
- Deployed improved challenger model

## Screenshots
(pending post latest model run 4.13.26)

## Key Engineering Challenges & Solutions

### 1. Stateful Data Pipeline Bugs (S3 Persistence)
**Problem:**  
Pipeline produced inconsistent results due to residual test data (`combined_test.csv`) persisting across runs.

**Impact:**  
- Incorrect baseline F1 scores  
- False drift detection  
- Output inconsistencies  

**Solution:**  
- Identified hidden state in S3 `/test/` and `/processed/` prefixes  
- Implemented controlled environment resets  
- Designed pipeline awareness of historical vs fresh runs  

---

### 2. Drift Detection Sensitivity Tuning
**Problem:**  
Changing drift threshold from `0.1 → 0.05` caused incorrect drift behavior.

**Impact:**  
- False positives in drift detection  
- Incorrect model retraining decisions  

**Solution:**  
- Restored threshold to `baseline_f1 - 0.1`  
- Validated using controlled dataset scenarios  
- Ensured alignment with evaluation expectations  

---

### 3. Champion/Challenger Evaluation Consistency
**Problem:**  
Mismatch between historical and current F1 scores.

**Impact:**  
- Incorrect deployment decisions  
- Pipeline logic inconsistencies  

**Solution:**  
- Ensured `old_test_set_f1` correctly persisted across runs  
- Fixed evaluation flow in Step Functions  
- Verified metric consistency across preprocessing + evaluation stages  

---

### 4. End-to-End Pipeline Debugging Across Services
**Problem:**  
Failures required tracing across:
- Lambda
- Step Functions
- SageMaker
- S3
- CloudWatch

**Solution:**  
- Used CloudWatch logs to trace data flow across services  
- Validated intermediate artifacts (S3 + logs)  
- Debugged distributed system behavior  


## Why This Project Matters

This project demonstrates:

- Designing and debugging distributed ML systems
- Handling stateful data pipelines in cloud environments
- Implementing automated retraining and deployment logic
- Working across multiple AWS services in production-like workflows
- Applying machine learning concepts (F1, drift detection) in real systems



## Future Improvements
- FastAPI inference layer
- Streamlit monitoring dashboard
- GitHub Actions CI/CD
- SNS alerts for drift and failed deployments
- Model registry integration

## Author
Tedra Birch |
University of Illinois Urbana - Champaign | 
Siebel School of Computing