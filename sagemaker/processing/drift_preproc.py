#!/usr/bin/env python3

import subprocess
import sys

libraries = ["scikit-learn", "pandas", "joblib", "boto3", "xgboost"]
subprocess.check_call([sys.executable, "-m", "pip", "install"] + libraries)

import os
import json
import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from utility import (
    download_s3_file,
    upload_s3_file,
    drop_unwanted_columns,
    local_inference_sklearn,
    get_production_model,
    preprocess_fraud_data
)

def parse_args():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-data-uri", type=str, required=True,
                        help="S3 URI for the raw input dataset")
    parser.add_argument("--test-data-uri", type=str, required=True,
                        help="S3 folder where the combined test set is stored (and will be overwritten).")
    parser.add_argument("--processed-data-uri", type=str, required=True,
                        help="S3 folder to store processed training data.")
    parser.add_argument("--file-name", type=str, required=True,
                        help="Name of the input file")
    parser.add_argument("--staging", type=str, required=True,
                        help="Name of the staging folder to store new preprocessor artifacts.")
    return parser.parse_args()


def main():
    args = parse_args()
    print("Arguments:")
    print(json.dumps(vars(args), indent=2))
    
    local_input_dir = "/opt/ml/processing/input"
    local_output_dir = "/opt/ml/processing/output"
    os.makedirs(local_input_dir, exist_ok=True)
    os.makedirs(local_output_dir, exist_ok=True)
    
    raw_local_file = os.path.join(local_input_dir, args.file_name)
    combined_test_local = os.path.join(local_input_dir, "combined_test.csv")
    processed_test_local = os.path.join(local_output_dir, "combined_test_processed.csv")
    processed_train_local = os.path.join(local_output_dir, f"processed_{args.file_name}")
    drift_result_path = os.path.join(local_output_dir, "drift_result.json")
    
    production_model, prod_imputer, prod_scaler = get_production_model()
    if production_model is not None and prod_imputer is not None and prod_scaler is not None:
        print("Production model loaded successfully with its imputer and scaler.")
        production_available = True
    else:
        print("Production model or its preprocessor artifacts not found. Forcing drift.")
        production_available = False

    print(f"Downloading raw data from {args.raw_data_uri} -> {raw_local_file}")
    download_s3_file(args.raw_data_uri, raw_local_file)
    
    combined_test_exists = True
    try:
        print(f"Downloading combined test data from {args.test_data_uri} -> {combined_test_local}")
        download_s3_file(os.path.join(args.test_data_uri, "combined_test.csv"), combined_test_local)
        print("Existing raw combined test data found.")
        raw_combined_test = pd.read_csv(combined_test_local)
        raw_combined_test = drop_unwanted_columns(raw_combined_test)
    except Exception as e:
        print("No raw combined test data found; this is the first run. Forcing drift detection.")
        combined_test_exists = False
        raw_combined_test = None

    df_new = pd.read_csv(raw_local_file)
    df_new = drop_unwanted_columns(df_new)
    print("New data columns after dropping unwanted columns:")
    print(df_new.columns.tolist())
    print(f"New data shape: {df_new.shape}")
    
    if df_new.shape[0] < 2:
        print("Not enough rows to split. Forcing drift.")
        drift_detected = True
        train_df = df_new.copy()
        test_df = pd.DataFrame()
    else:
        train_df, test_df = train_test_split(df_new, test_size=0.2, random_state=42)
    
    baseline_f1 = 0
    combined_f1 = 0

    if not combined_test_exists:
        print("Storing new test data as the raw combined test data.")
        test_df.to_csv(combined_test_local, index=False)
        raw_test_s3_uri = os.path.join(args.test_data_uri, "combined_test.csv")
        upload_s3_file(combined_test_local, raw_test_s3_uri)
        print(f"Uploaded raw combined test data to: {raw_test_s3_uri}")
        
        if production_available:
            processed_new, _, _ = preprocess_fraud_data(test_df, prod_imputer, prod_scaler, is_training=False)
        else:
            processed_new = test_df.copy()
        processed_new.to_csv(processed_test_local, index=False)
        drift_detected = True
        updated_raw = test_df.copy()
    else:
        if production_available:
            baseline_processed, _, _ = preprocess_fraud_data(raw_combined_test, prod_imputer, prod_scaler, is_training=False)
            temp_model_path = os.path.join("/tmp", "production_model_temp.joblib")
            joblib.dump(production_model, temp_model_path)
            baseline_metrics = local_inference_sklearn(temp_model_path, baseline_processed, label_col="is_fraud")
            baseline_f1 = baseline_metrics.get("f1", 0)
            print("Baseline f1 on historical test data:", baseline_f1)
            if not test_df.empty:
                new_test_processed, _, _ = preprocess_fraud_data(test_df, prod_imputer, prod_scaler, is_training=False)
                combined_processed = pd.concat([baseline_processed, new_test_processed], ignore_index=True)
                combined_metrics = local_inference_sklearn(temp_model_path, combined_processed, label_col="is_fraud")
                combined_f1 = combined_metrics.get("f1", 0)
                print("Combined f1 (historical + new):", combined_f1)
                drift_detected = combined_f1 < baseline_f1 - 0.1
            else:
                print("No new test data available; forcing drift.")
                drift_detected = True
            updated_raw = pd.concat([raw_combined_test, test_df], ignore_index=True)
        else:
            drift_detected = True
            updated_raw = pd.concat([raw_combined_test, test_df], ignore_index=True) if raw_combined_test is not None else test_df

    if drift_detected:
        print("Drift detected. Re-training preprocessor using new training data and creating new artifacts.")
        new_train_processed, new_imputer, new_scaler = preprocess_fraud_data(train_df, is_training=True)
        new_train_processed.to_csv(processed_train_local, index=False)
        
        if raw_combined_test is not None:
            new_hist_processed, _, _ = preprocess_fraud_data(raw_combined_test, new_imputer, new_scaler, is_training=False)
        else:
            new_hist_processed = pd.DataFrame()
        if not test_df.empty:
            new_new_processed, _, _ = preprocess_fraud_data(test_df, new_imputer, new_scaler, is_training=False)
        else:
            new_new_processed = pd.DataFrame()
        
        if not new_hist_processed.empty and not new_new_processed.empty:
            final_processed = pd.concat([new_hist_processed, new_new_processed], ignore_index=True)
        elif not new_hist_processed.empty:
            final_processed = new_hist_processed
        else:
            final_processed = new_new_processed
        print("Final processed combined test data shape (using new preprocessor):", final_processed.shape)
        final_processed.to_csv(processed_test_local, index=False)
        
        updated_raw.to_csv(combined_test_local, index=False)
        raw_test_s3_uri = os.path.join(args.test_data_uri, "combined_test.csv")
        upload_s3_file(combined_test_local, raw_test_s3_uri)
        print(f"Uploaded processed combined test to: {raw_test_s3_uri}")
        
        processed_test_s3_uri = os.path.join(args.test_data_uri, "combined_test_processed.csv")
        upload_s3_file(processed_test_local, processed_test_s3_uri)
        print(f"Uploaded processed combined test to: {processed_test_s3_uri}")
        
        processed_train_s3_uri = os.path.join(args.processed_data_uri, f"processed_{args.file_name}").replace("\\", "/")
        upload_s3_file(processed_train_local, processed_train_s3_uri)
        print(f"Uploaded processed training data to: {processed_train_s3_uri}")
        
        staging_local = "/tmp/staging"
        os.makedirs(staging_local, exist_ok=True)
        staging_imputer_local = os.path.join(staging_local, "imputer.pkl")
        staging_scaler_local = os.path.join(staging_local, "scaler.pkl")
        joblib.dump(new_imputer, staging_imputer_local)
        joblib.dump(new_scaler, staging_scaler_local)
        upload_s3_file(staging_imputer_local, os.path.join(args.staging, "imputer.pkl"))
        upload_s3_file(staging_scaler_local, os.path.join(args.staging, "scaler.pkl"))
        print(f"New artifacts uploaded to staging S3 folder: {args.staging}")
    else:
        print("No drift detected. Applying minimal transformation using production preprocessor.")
        train_df.to_csv(processed_train_local, index=False)
        if combined_test_exists and production_available:
            processed_hist = preprocess_fraud_data(raw_combined_test, prod_imputer, prod_scaler, is_training=False)[0]
        else:
            processed_hist = pd.DataFrame()
        if not test_df.empty and production_available:
            processed_new_test = preprocess_fraud_data(test_df, prod_imputer, prod_scaler, is_training=False)[0]
        else:
            processed_new_test = pd.DataFrame()
        final_processed = pd.concat([processed_hist, processed_new_test], ignore_index=True)
        final_processed.to_csv(processed_test_local, index=False)
        
        processed_test_s3_uri = os.path.join(args.test_data_uri, "combined_test_processed.csv")
        upload_s3_file(processed_test_local, processed_test_s3_uri)
        print(f"Uploaded processed combined test to: {processed_test_s3_uri}")
        
        processed_train_s3_uri = os.path.join(args.processed_data_uri, f"processed_{args.file_name}").replace("\\", "/")
        upload_s3_file(processed_train_local, processed_train_s3_uri)
        print(f"Uploaded processed training data to: {processed_train_s3_uri}")
    
    drift_result = {"drift_detected": bool(drift_detected), "old_test_set_f1": baseline_f1, "new_test_set_f1": combined_f1}
    with open(drift_result_path, "w") as f:
        json.dump(drift_result, f)

if __name__ == "__main__":
    main()
