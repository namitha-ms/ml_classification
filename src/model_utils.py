import os
import json
from typing import Dict, Any
import numpy as np
import joblib
from sklearn.metrics import (
    accuracy_score,
    roc_auc_score,
    precision_score,
    recall_score,
    f1_score,
    matthews_corrcoef
)


def evaluate_model(model, X_test: np.ndarray, y_test: np.ndarray) -> Dict[str, float]:
    # Generate predictions
    y_pred = model.predict(X_test)
    
    # Get probability predictions for AUC score
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    
    # Calculate all six metrics
    metrics = {
        'accuracy': accuracy_score(y_test, y_pred),
        'auc_score': roc_auc_score(y_test, y_pred_proba),
        'precision': precision_score(y_test, y_pred, zero_division=0),
        'recall': recall_score(y_test, y_pred, zero_division=0),
        'f1_score': f1_score(y_test, y_pred, zero_division=0),
        'mcc_score': matthews_corrcoef(y_test, y_pred)
    }
    
    return metrics


def save_model(model, model_name: str, output_dir: str = "model/") -> None:

    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Construct full file path
    file_path = os.path.join(output_dir, f"{model_name}.pkl")
    
    # Save model using joblib
    try:
        joblib.dump(model, file_path)
    except Exception as e:
        raise IOError(f"Failed to save model to '{file_path}'. Error: {str(e)}")


def load_model(model_name: str, model_dir: str = "model/"):
    # Construct full file path
    file_path = os.path.join(model_dir, f"{model_name}.pkl")
    
    # Check if file exists
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Model file not found at '{file_path}'")
    
    # Load model using joblib
    try:
        model = joblib.load(file_path)
        return model
    except Exception as e:
        raise IOError(f"Failed to load model from '{file_path}'. Error: {str(e)}")


def save_metrics(metrics_dict: Dict[str, Dict[str, float]], 
                 output_path: str = "model/metrics.json") -> None:
    
    # Create output directory if it doesn't exist
    output_dir = os.path.dirname(output_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    
    # Save metrics to JSON
    try:
        with open(output_path, 'w') as f:
            json.dump(metrics_dict, f, indent=2)
    except Exception as e:
        raise IOError(f"Failed to save metrics to '{output_path}'. Error: {str(e)}")


def load_metrics(metrics_path: str = "model/metrics.json") -> Dict[str, Dict[str, float]]:
    # Check if file exists
    if not os.path.exists(metrics_path):
        raise FileNotFoundError(f"Metrics file not found at '{metrics_path}'")
    
    # Load metrics from JSON
    try:
        with open(metrics_path, 'r') as f:
            metrics = json.load(f)
        return metrics
    except Exception as e:
        raise IOError(f"Failed to load metrics from '{metrics_path}'. Error: {str(e)}")
