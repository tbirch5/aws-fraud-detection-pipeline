#!/usr/bin/env python3
import os
import json
import argparse
import joblib
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score
from utility import download_s3_file, extract_tarball, get_production_model, local_inference_sklearn, preprocess_fraud_data
import subprocess
import sys

subprocess.check_call([sys.executable, "-m", "pip", "install", "xgboost"])

def evaluate_model(model, X, y):
    # Compute predictions
    y_pred = model.predict(X)
    # Use threshold 0.5 for binary classification
    y_pred_binary = (y_pred > 0.5).astype(int)
    acc = accuracy_score(y, y_pred_binary)
    f1 = f1_score(y, y_pred_binary)
    return {"accuracy": acc, "f1": f1}

def parse_args():
    parser = argparse.ArgumentParser(description="Champion vs. Challenger Evaluation Script")
    parser.add_argument("challenger_model_file", type=str,
                        help="S3 URI to the challenger model artifact (tar.gz)")
    parser.add_argument("test_data_uri", type=str,
                        help="S3 URI for evaluation data (informational)")
    parser.add_argument("--test-file", type=str, default="/opt/ml/processing/input/data/combined_test_processed.csv",
                        help="Local path to the test CSV file")
    parser.add_argument("--output-dir", type=str, default="/opt/ml/processing/output",
                        help="Local path to store evaluation results")
    return parser.parse_args()

def main():
    args = parse_args()
    print("Arguments:")
    print(json.dumps(vars(args), indent=2))
    
    # --- Load and extract challenger model artifacts ---
    local_challenger_tar = "/tmp/challenger_model.tar.gz"
    print(f"Downloading challenger model from {args.challenger_model_file} to {local_challenger_tar} ...")
    download_s3_file(args.challenger_model_file, local_challenger_tar)
    
    local_challenger_dir = "/tmp/challenger_model"
    os.makedirs(local_challenger_dir, exist_ok=True)
    extract_tarball(local_challenger_tar, local_challenger_dir)
    
    # Load challenger model from tarball.
    challenger_model_path = os.path.join(local_challenger_dir, "model.joblib")
    if not os.path.exists(challenger_model_path):
        raise FileNotFoundError(f"Challenger model not found at {challenger_model_path}")
    print(f"Loading challenger model from {challenger_model_path} ...")
    challenger_model = joblib.load(challenger_model_path)
    
    # Load challenger preprocessor artifacts.
    challenger_imputer_path = os.path.join(local_challenger_dir, "imputer.pkl")
    challenger_scaler_path = os.path.join(local_challenger_dir, "scaler.pkl")
    if not os.path.exists(challenger_imputer_path) or not os.path.exists(challenger_scaler_path):
        raise FileNotFoundError("Challenger preprocessor artifacts (imputer and scaler) not found in tarball.")
    challenger_imputer = joblib.load(challenger_imputer_path)
    challenger_scaler = joblib.load(challenger_scaler_path)
    
    # --- Load champion model artifacts via get_production_model() ---
    champion_model, champ_imputer, champ_scaler = get_production_model()
    if champion_model is not None and champ_imputer is not None and champ_scaler is not None:
        print("Champion model and its preprocessor loaded successfully.")
    else:
        print("No champion model or its preprocessor artifacts found. Setting champion metrics to zero.")
    
    # --- Load test data ---
    df_test = pd.read_csv(args.test_file)
    if "is_fraud" not in df_test.columns:
        raise ValueError("Test data must contain an 'is_fraud' column.")
    y_test = df_test["is_fraud"]
    
    # --- Preprocess test data using challenger's pipeline ---
    challenger_test, _, _ = preprocess_fraud_data(df_test, challenger_imputer, challenger_scaler, is_training=False)
    
    # --- Evaluate challenger model ---
    print("Evaluating challenger model on test data ...")
    # Drop label column from features when evaluating
    challenger_results = evaluate_model(challenger_model, challenger_test.drop("is_fraud", axis=1), y_test)
    print("Challenger Model Results:", challenger_results)
    
    # --- Evaluate champion model (if available) ---
    if champion_model is not None and champ_imputer is not None and champ_scaler is not None:
        champion_test, _, _ = preprocess_fraud_data(df_test, champ_imputer, champ_scaler, is_training=False)
        print("Evaluating champion model on test data ...")
        champion_results = evaluate_model(champion_model, champion_test.drop("is_fraud", axis=1), y_test)
        print("Champion Model Results:", champion_results)
        champion_f1 = champion_results.get("f1", 0)
    else:
        champion_results = {"accuracy": 0, "f1": 0}
        champion_f1 = 0
        print("No champion model available; using champion metrics as zero.")
    
    evaluation_result = {"champion_f1": champion_f1, "challenger_f1": challenger_results.get("f1", 0)}
    
    os.makedirs(args.output_dir, exist_ok=True)
    evaluation_output_path = os.path.join(args.output_dir, "evaluation.json")
    with open(evaluation_output_path, "w") as f:
        json.dump(evaluation_result, f)
    
    print("Evaluation completed successfully.")
    print("Evaluation result:", json.dumps(evaluation_result, indent=2))

if __name__ == "__main__":
    main()
