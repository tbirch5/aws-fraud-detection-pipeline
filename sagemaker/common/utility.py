#!/usr/bin/env python3

import os
import json
import boto3
import tarfile
import joblib
import pandas as pd
import numpy as np
import argparse
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer

def download_s3_file(s3_uri, local_path):
    """Download an S3 file to a local path."""
    s3 = boto3.client("s3")
    prefix_removed = s3_uri.replace("s3://", "")
    bucket, key = prefix_removed.split("/", 1)
    os.makedirs(os.path.dirname(local_path), exist_ok=True)
    s3.download_file(bucket, key, local_path)

def upload_s3_file(local_path, s3_uri):
    """Upload a local file to S3."""
    s3 = boto3.client("s3")
    prefix_removed = s3_uri.replace("s3://", "")
    bucket, key = prefix_removed.split("/", 1)
    s3.upload_file(local_path, bucket, key)

def drop_unwanted_columns(df):
    """
    Drops any unnamed index columns and the 'id' column if present.
    """
    df = df.copy()
    df = df.loc[:, ~df.columns.str.startswith("Unnamed")]
    if "id" in df.columns:
        df = df.drop(columns=["id"])
    return df

def local_inference_sklearn(model_path, test_df, label_col="is_fraud"):
    """
    Loads a scikit-learn model from the given path and computes fraud detection metrics.
    """
    model = joblib.load(model_path)
    if label_col not in test_df.columns:
        print(f"Warning: {label_col} not found, returning 0.0 for all metrics.")
        return {"accuracy": 0.0, "f1": 0.0, "precision": 0.0, "recall": 0.0}
    X_test = test_df.drop(label_col, axis=1)
    y_test = test_df[label_col]
    preds = model.predict(X_test)
    metrics = {
        "accuracy": accuracy_score(y_test, preds),
        "f1": f1_score(y_test, preds),
        "precision": precision_score(y_test, preds),
        "recall": recall_score(y_test, preds)
    }
    return metrics

def create_model_tarball(model_dir, tarball_path, files_to_include):
    """
    Create a tar.gz archive from model_dir containing the specified files.
    
    Parameters:
      model_dir (str): Directory where the model files reside.
      tarball_path (str): Full path to the output tar.gz archive.
      files_to_include (list): List of filenames to include in the tarball.
    """
    with tarfile.open(tarball_path, "w:gz") as tar:
        for filename in files_to_include:
            file_path = os.path.join(model_dir, filename)
            if os.path.exists(file_path):
                tar.add(file_path, arcname=filename)
            else:
                print(f"Warning: {filename} not found in {model_dir}")

def extract_tarball(tarball_path, extract_to):
    """
    Extracts a tar.gz archive to the specified directory.
    """
    print(f"Extracting {tarball_path} to {extract_to}...")
    with tarfile.open(tarball_path, "r:gz") as tar:
        tar.extractall(path=extract_to)
    return extract_to

def get_production_model(model_name="mp13-production-model"):
    """
    Retrieves and loads the production (champion) model with the specified model name from SageMaker.
    
    The function calls the SageMaker describe_model API to get model details,
    obtains the ModelDataUrl from the PrimaryContainer, downloads the associated
    tar.gz artifact (which should include a 'model.joblib' file, along with
    'imputer.pkl' and 'scaler.pkl'), extracts it, and loads the model using joblib.
    
    Returns:
       tuple: (model, imputer, scaler) or (None, None, None) if any step fails.
    """
    region = os.environ.get("AWS_REGION", "us-east-1")
    sagemaker_client = boto3.client("sagemaker", region_name=region)
    
    try:
        response = sagemaker_client.describe_model(ModelName=model_name)
    except Exception as e:
        print(f"Error describing model {model_name}: {e}")
        return None, None, None

    container = response.get("PrimaryContainer", {})
    if "ModelDataUrl" not in container:
        print("Production model container does not have a ModelDataUrl.")
        return None, None, None

    model_data_url = container["ModelDataUrl"]
    print("Production model artifact S3 URI:", model_data_url)
    
    try:
        local_tar_path = "/tmp/production_model.tar.gz"
        download_s3_file(model_data_url, local_tar_path)
    except Exception as e:
        print(f"Error describing model {model_name}: {e}")
        return None, None, None
    
    local_extract_dir = "/tmp/production_model"
    os.makedirs(local_extract_dir, exist_ok=True)
    extract_tarball(local_tar_path, local_extract_dir)
    
    local_model_path = os.path.join(local_extract_dir, "model.joblib")
    if not os.path.exists(local_model_path):
        print("Model file not found in extracted artifact.")
        return None, None, None

    model = joblib.load(local_model_path)
    
    local_imputer_path = os.path.join(local_extract_dir, "imputer.pkl")
    local_scaler_path  = os.path.join(local_extract_dir, "scaler.pkl")
    
    if os.path.exists(local_imputer_path):
        imputer = joblib.load(local_imputer_path)
    else:
        print("Imputer file not found in extracted artifact.")
        imputer = None

    if os.path.exists(local_scaler_path):
        scaler = joblib.load(local_scaler_path)
    else:
        print("Scaler file not found in extracted artifact.")
        scaler = None

    return model, imputer, scaler


def preprocess_fraud_data(df, imputer=None, scaler=None, is_training=True):
    df = df.copy()
    if "is_fraud" in df.columns:
        nan_count = df["is_fraud"].isna().sum()
        if nan_count > 0:
            print(f"WARNING: Found {nan_count} NaNs in 'is_fraud'. Dropping those rows.")
            df = df.dropna(subset=["is_fraud"])
            df["is_fraud"] = df["is_fraud"].astype(int)
    if "trans_date_trans_time" in df.columns:
        df["trans_date_trans_time"] = pd.to_datetime(df["trans_date_trans_time"], errors="coerce")
        df["hour"] = df["trans_date_trans_time"].dt.hour
        df["day_of_week"] = df["trans_date_trans_time"].dt.dayofweek
        df["month"] = df["trans_date_trans_time"].dt.month
        df["transaction_day"] = df["trans_date_trans_time"].dt.day
        df["is_weekend"] = df["day_of_week"].isin([5, 6]).astype(int)
        df = df.drop(columns=["trans_date_trans_time"])
    if all(col in df.columns for col in ["lat", "long", "merch_lat", "merch_long"]):
        df["distance"] = ((df["lat"] - df["merch_lat"])**2 + (df["long"] - df["merch_long"])**2)**0.5
        df["amt_to_distance_ratio"] = df["amt"] / (df["distance"] + 1)
    if "is_fraud" in df.columns:
        X = df.drop("is_fraud", axis=1)
        y = df["is_fraud"]
    else:
        X = df
        y = None
    categorical_cols = X.select_dtypes(include=["object"]).columns
    high_card_cols = [col for col in categorical_cols if X[col].nunique() > 100]
    X = X.drop(columns=high_card_cols)
    remaining_cats = [col for col in categorical_cols if col not in high_card_cols]
    X = pd.get_dummies(X, columns=remaining_cats, drop_first=True)
    X = X.replace([float("inf"), float("-inf")], pd.NA)
    if is_training:
        print("Training mode: fitting imputer and scaler.")
        imputer = SimpleImputer(strategy="mean")
        X_imputed = imputer.fit_transform(X)
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X_imputed)
    else:
        if imputer is None or scaler is None:
            raise ValueError("For testing, a production imputer and scaler must be provided.")
        expected_columns = imputer.feature_names_in_ if hasattr(imputer, "feature_names_in_") else None
        if expected_columns is not None:
            current_cols = X.columns.tolist()
            X_aligned = pd.DataFrame(0, index=X.index, columns=expected_columns)
            for col in current_cols:
                if col in expected_columns:
                    X_aligned[col] = X[col]
            X = X_aligned
        X_imputed = imputer.transform(X)
        X_scaled = scaler.transform(X_imputed)
    X_processed = pd.DataFrame(X_scaled, columns=X.columns)
    if y is not None:
        y = y.astype(int)
        X_processed["is_fraud"] = y.values
        if X_processed["is_fraud"].isna().sum() > 0:
            print("WARNING: NaN values found in 'is_fraud' after merging. Dropping these rows.")
            X_processed = X_processed.dropna(subset=["is_fraud"])
            X_processed["is_fraud"] = X_processed["is_fraud"].astype(int)
    print("Preprocessing complete!")
    return X_processed, imputer, scaler

# def get_production_model(model_package_group):
#     """
#     Retrieves and loads the current production (champion) model for the specified 
#     model package group from the SageMaker Model Registry.
    
#     This function queries the registry for the latest approved model package,
#     downloads the associated model artifact (assumed to be a tar.gz archive with a 'model.joblib'
#     file), extracts it, and loads the model using joblib.
    
#     Returns the loaded model, or None if no approved model exists.
#     """
#     region = os.environ.get("AWS_REGION", "us-east-1")
#     sagemaker_client = boto3.client("sagemaker", region_name=region)
#     response = sagemaker_client.list_model_packages(
#         ModelPackageGroupName=model_package_group,
#         ModelApprovalStatus="Approved",
#         SortBy="CreationTime",
#         SortOrder="Descending",
#         MaxResults=1
#     )
#     summaries = response.get("ModelPackageSummaryList", [])
#     if not summaries:
#         print("No approved model package found.")
#         return None

#     model_package_arn = summaries[0]["ModelPackageArn"]
#     details = sagemaker_client.describe_model_package(ModelPackageName=model_package_arn)
#     containers = details.get("InferenceSpecification", {}).get("Containers", [])
#     if not containers or "ModelDataUrl" not in containers[0]:
#         print("Approved model package found, but no ModelDataUrl provided.")
#         return None

#     model_data_url = containers[0]["ModelDataUrl"]
#     print("Production model artifact S3 URI:", model_data_url)
#     local_tar_path = "/tmp/production_model.tar.gz"
#     download_s3_file(model_data_url, local_tar_path)
#     local_extract_dir = "/tmp/production_model"
#     os.makedirs(local_extract_dir, exist_ok=True)
#     extract_tarball(local_tar_path, local_extract_dir)
#     local_model_path = os.path.join(local_extract_dir, "model.joblib")
#     if not os.path.exists(local_model_path):
#         print("Model file not found in extracted artifact.")
#         return None
#     return joblib.load(local_model_path)
