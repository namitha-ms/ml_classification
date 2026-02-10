import os
import pandas as pd
import numpy as np
import joblib
from typing import Tuple
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


FEATURE_COLUMNS = [
    'LIMIT_BAL',
    'SEX',
    'EDUCATION',
    'MARRIAGE',
    'AGE',
    'PAY_1',
    'PAY_2',
    'PAY_3',
    'PAY_4',
    'PAY_5',
    'PAY_6',
    'BILL_AMT1',
    'BILL_AMT2',
    'BILL_AMT3',
    'BILL_AMT4',
    'BILL_AMT5',
    'BILL_AMT6',
    'PAY_AMT1',
    'PAY_AMT2',
    'PAY_AMT3',
    'PAY_AMT4',
    'PAY_AMT5',
    'PAY_AMT6',
]

TARGET_COLUMN = 'dpnm'

SCALER_PATH = "model/scaler.pkl"


def load_dataset(data_path: str = "data/default of credit card clients.csv") -> Tuple[pd.DataFrame, pd.Series]:
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Dataset file not found at '{data_path}'.")
    
    try:
        df = pd.read_csv(data_path)
    except Exception as e:
        raise ValueError(f"Failed to read CSV file. Error: {str(e)}")
    
    if df.empty:
        raise ValueError("Dataset file is empty.")
    
    if 'ID' in df.columns:
        df = df.drop(columns=['ID'])
    
    required_columns = FEATURE_COLUMNS + [TARGET_COLUMN]
    missing_columns = [col for col in required_columns if col not in df.columns]
    
    if missing_columns:
        raise ValueError(f"Dataset is missing required columns: {missing_columns}")
    
    features = df[FEATURE_COLUMNS].copy()
    target = df[TARGET_COLUMN].copy()
    
    return features, target


def count_class_distribution(target: pd.Series) -> dict:

    value_counts = target.value_counts()
    negative_count = value_counts.get(0, 0)
    positive_count = value_counts.get(1, 0)
    total = negative_count + positive_count
    
    scale_pos_weight = negative_count / positive_count if positive_count > 0 else 0
    
    return {
        'negative_class (0)': negative_count,
        'positive_class (1)': positive_count,
        'total': total,
        'negative_ratio': negative_count / total if total > 0 else 0,
        'positive_ratio': positive_count / total if total > 0 else 0,
        'scale_pos_weight': round(scale_pos_weight, 2)
    }


def preprocess_data(
    features: pd.DataFrame,
    target: pd.Series,
    test_size: float = 0.2,
    random_state: int = 42,
    save_test_data: bool = True
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:

    # Split data first 
    X_train, X_test, y_train, y_test = train_test_split(
        features,
        target,
        test_size=test_size,
        random_state=random_state
    )
    
    # Scale features using StandardScaler
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Save the scaler for use in the Streamlit app
    joblib.dump(scaler, SCALER_PATH)
    print(f"Scaler saved to {SCALER_PATH}")
    
    # Convert target to numpy arrays
    y_train_arr = y_train.values
    y_test_arr = y_test.values
    
    # Save unscaled test data with labels for download
    if save_test_data:
        test_data_df = X_test.copy()
        test_data_df[TARGET_COLUMN] = y_test.values
        test_data_df.to_csv("data/test_data.csv", index=False)
        print("Test data saved to data/test_data.csv")
    
    return X_train_scaled, X_test_scaled, y_train_arr, y_test_arr
