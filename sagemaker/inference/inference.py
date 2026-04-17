#!/usr/bin/env python3
import os
import sys
import json
import joblib
import numpy as np
import subprocess

# Ensure that xgboost is installed
subprocess.check_call([sys.executable, "-m", "pip", "install", "xgboost"])
import xgboost as xgb

def model_fn(model_dir):
    model_path = os.path.join(model_dir, "model.joblib")
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Fraud detection model not found at: {model_path}")
    print(f"Loading fraud detection model from {model_path}")
    return joblib.load(model_path)

def input_fn(request_body, content_type):
    if content_type == "application/json":
        data = json.loads(request_body)
        if isinstance(data, dict):
            # Single prediction wrapped as a 2D array
            return np.array([list(data.values())])
        elif isinstance(data, list):
            if all(isinstance(item, dict) for item in data):
                # Batch prediction from a list of dictionaries
                return np.array([list(item.values()) for item in data])
            else:
                # If already a list of feature lists
                return np.array(data)
        else:
            raise ValueError("Input data must be a dictionary or a list")
    else:
        raise ValueError(f"Unsupported content type: {content_type}. Only application/json is supported.")

def predict_fn(input_data, model):
    raw_predictions = model.predict(input_data)
    binary_predictions = (raw_predictions > 0.5).astype(int)
    return {
        "probability": raw_predictions.tolist(),
        "prediction": binary_predictions.tolist()
    }

def output_fn(prediction_dict, accept):
    if accept == "application/json":
        return json.dumps(prediction_dict)
    else:
        raise ValueError(f"Unsupported accept type: {accept}. Only application/json is supported.")

if __name__ == "__main__":
    # This main block is for local testing only.
    if len(sys.argv) != 3:
        print("Usage: inference.py <model_dir> <json_request_body>")
        sys.exit(1)
    model_directory = sys.argv[1]
    request_body = sys.argv[2]
    
    model = model_fn(model_directory)
    input_data = input_fn(request_body, "application/json")
    predictions = predict_fn(input_data, model)
    output = output_fn(predictions, "application/json")
    print(output)
