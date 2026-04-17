#!/usr/bin/env python3
import sys
import subprocess
subprocess.check_call([sys.executable, "-m", "pip", "install", "xgboost"])
import os
import pandas as pd
import joblib
import shutil
import xgboost as xgb
from utility import get_production_model, create_model_tarball, download_s3_file, upload_s3_file

def train_model(X, y, existing_model=None):
    xgb_params = {
        "objective": "binary:logistic",
        "eval_metric": "logloss",
        "learning_rate": 0.1,
        "max_depth": 6,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "min_child_weight": 1,
        "scale_pos_weight": 1,
        "random_state": 42
    }
    
    model = xgb.XGBClassifier(**xgb_params)
    
    if existing_model is not None:
        print("Warm-starting training with the existing production model...")
        model.fit(X, y, xgb_model=existing_model)
    else:
        print("Training a new model from scratch...")
        model.fit(X, y)
        
    return model

def main():
    if len(sys.argv) < 3:
        print("Usage: python3 train.py <processed_filename> <staging folder>")
        sys.exit(1)
    file_name = sys.argv[1]
    staging_folder = sys.argv[2]
    
    # Load processed training data
    data_path = os.path.join("/opt/ml/input/data/train", file_name)
    print(f"Loading data from: {data_path}")
    if not os.path.exists(data_path):
        print(f"Error: File not found at {data_path}")
        sys.exit(1)
    data = pd.read_csv(data_path)
    print(f"Loaded data shape: {data.shape}")
    if "is_fraud" not in data.columns:
        print("Error: 'is_fraud' column not found in data.")
        sys.exit(1)
    y = data["is_fraud"]
    X = data.drop("is_fraud", axis=1)
    print(f"Features shape: {X.shape}, Target shape: {y.shape}")
    
    # Try to obtain production model (returns model, imputer, scaler)
    existing_model, prod_imputer, prod_scaler = get_production_model()
    if existing_model is not None:
        print("Production model found; using it for warm-start training.")
    else:
        print("No production model found; training from scratch.")
    
    # Train model (warm-start if production model exists)
    model = train_model(X, y, existing_model=existing_model)
    
    # Store the trained model in the output directory.
    output_dir = "/opt/ml/model"
    os.makedirs(output_dir, exist_ok=True)
    model_file_path = os.path.join(output_dir, "model.joblib")
    joblib.dump(model, model_file_path)
    print(f"Model saved to {model_file_path}")
    
    # Copy inference.py from the script folder.
    source_inference = os.path.join("/opt/ml/input/data/script", "inference.py")
    dest_inference = os.path.join(output_dir, "inference.py")
    if os.path.exists(source_inference):
        shutil.copy(source_inference, dest_inference)
        print(f"Copied inference.py from {source_inference} to {dest_inference}")
    else:
        print(f"Warning: inference.py not found at {source_inference}.")
    
    # Retrieve imputer and scaler from the staging folder (which is an S3 URI).
    # Download them locally.
    staging_imputer_s3 = os.path.join(staging_folder, "imputer.pkl")
    staging_scaler_s3 = os.path.join(staging_folder, "scaler.pkl")
    local_imputer_path = os.path.join("/tmp", "imputer.pkl")
    local_scaler_path = os.path.join("/tmp", "scaler.pkl")
    try:
        download_s3_file(staging_imputer_s3, local_imputer_path)
        download_s3_file(staging_scaler_s3, local_scaler_path)
        print(f"Downloaded imputer and scaler from staging S3: {staging_imputer_s3}, {staging_scaler_s3}")
        dest_imputer_path = os.path.join(output_dir, "imputer.pkl")
        dest_scaler_path = os.path.join(output_dir, "scaler.pkl")
        shutil.copy(local_imputer_path, dest_imputer_path)
        shutil.copy(local_scaler_path, dest_scaler_path)
        print(f"Copied imputer and scaler to output directory: {dest_imputer_path}, {dest_scaler_path}")
    except Exception as e:
        print("Warning: Could not download imputer/scaler from staging S3 folder:", e)
    
    # Create a tarball that bundles the model artifact.
    tarball_path = os.path.join(output_dir, "model.tar.gz")
    files_to_include = ["model.joblib", "inference.py"]
    if os.path.exists(os.path.join(output_dir, "imputer.pkl")):
        files_to_include.append("imputer.pkl")
    if os.path.exists(os.path.join(output_dir, "scaler.pkl")):
        files_to_include.append("scaler.pkl")
    create_model_tarball(output_dir, tarball_path, files_to_include)
    print(f"Model artifact tarball created at {tarball_path}")
    print("Training job completed.")

if __name__ == "__main__":
    main()
