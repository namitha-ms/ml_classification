"""
Model validation script.
Loads all trained models from disk and verifies they can make predictions.
"""

import pandas as pd
from src.models import (
    LogisticRegressionClf,
    DecisionTreeClf,
    KNNClf,
    NaiveBayesClf,
    RandomForestClf,
    XGBoostClf
)
from src.model_utils import load_metrics
from src.data_processing import FEATURE_COLUMNS, TARGET_COLUMN


def main():
    # Load test data (contains features and labels)
    print("Loading test data...")
    test_data = pd.read_csv('data/test_data.csv')
    
    # Extract features and labels
    test_features = test_data[FEATURE_COLUMNS]
    test_labels = test_data[TARGET_COLUMN] if TARGET_COLUMN in test_data.columns else None

    print(f"Test features shape: {test_features.shape}")
    if test_labels is not None:
        print(f"Test labels shape: {test_labels.shape}")
    else:
        print("No labels found in test data")

    # Test loading each model and making predictions
    models = [
        LogisticRegressionClf(),
        DecisionTreeClf(),
        KNNClf(),
        NaiveBayesClf(),
        RandomForestClf(),
        XGBoostClf()
    ]

    print("\nTesting model loading and predictions:")
    for model in models:
        try:
            model.load()
            predictions = model.predict(test_features.values)
            print(f"{model.model_name}: loaded successfully, predictions shape: {predictions.shape}")
        except Exception as e:
            print(f"{model.model_name}: FAILED - {str(e)}")

    # Load and verify metrics
    print("\nLoading metrics...")
    try:
        metrics = load_metrics()
        print(f"Metrics loaded successfully for models: {list(metrics.keys())}")
    except Exception as e:
        print(f"Failed to load metrics: {str(e)}")

    print("\n" + "=" * 50)
    print("Model validation complete!")
    print("=" * 50)


if __name__ == "__main__":
    main()
