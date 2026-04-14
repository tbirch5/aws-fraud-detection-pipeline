# Fraud Detection MLOps Pipeline on AWS

## Overview
This is an implementation of an end-to-end cloud-native machine learning pipeline for fraud detection using AWS Step Functions, AWS Lambda, Amazon SageMaker, and Amazon S3. The pipeline automates data preprocessing, drift checks, model retraining, evaluation, champion/challenger comparison, and deployment.

## Problem Statement
Fraud detection systems must adapt to changing transaction behavior over time. This pipeline addresses that challenge by orchestrating automated model retraining and deployment decisions based on evaluation metrics and drift-aware workflow logic.

## Architecture
(architecture diagram pending)

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
(pending overview of folders)

## Results
- Automated retraining pipeline
- Model evaluation artifacts
- Champion/challenger selection
- Successful endpoint deployment

## Screenshots
(pending post latest model run 4.13.26)

## Future Improvements
- FastAPI inference layer
- Streamlit monitoring dashboard
- GitHub Actions CI/CD
- SNS alerts for drift and failed deployments
- Model registry integration

## Author
Tedra Birch
University of Illinois Urbana - Champaign
Siebel School of Computing