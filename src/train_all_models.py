"""
Master training script for all classification models.
Loads data, trains all six models, evaluates them, and saves artifacts.
"""

from typing import Dict, Type
import numpy as np

from src.data_processing import load_dataset, preprocess_data, count_class_distribution
from src.models import LogisticRegressionClf, DecisionTreeClf, KNNClf, NaiveBayesClf, RandomForestClf, XGBoostClf
from src.models.base_model import BaseClassifier
from src.model_utils import save_metrics


# Model classes with their display names
MODEL_CLASSES: Dict[str, Type[BaseClassifier]] = {
    "Logistic Regression": LogisticRegressionClf,
    "Decision Tree": DecisionTreeClf,
    "K-Nearest Neighbors": KNNClf,
    "Naive Bayes": NaiveBayesClf,
    "Random Forest": RandomForestClf,
    "XGBoost": XGBoostClf,
}


def train_and_evaluate_model(
    model_class: Type[BaseClassifier],
    display_name: str,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray
) -> tuple:

    print("\n" + "=" * 50)
    print(f"Training {display_name}...")
    print("=" * 50)
    
    model = model_class()
    model.train(X_train, y_train)
    metrics = model.evaluate(X_test, y_test)
    model.save()
    
    print(f"\n{display_name} Metrics:")
    for metric_name, value in metrics.items():
        print(f"  {metric_name}: {value:.4f}")
    
    return model.model_name, metrics


def main():
    # Load and preprocess data
    print("Loading dataset...")
    features, target = load_dataset("data/credit_card_default.csv")
    print(f"Dataset loaded: {features.shape[0]} samples, {features.shape[1]} features")
    
    # Print class distribution
    print("\nClass Distribution:")
    distribution = count_class_distribution(target)
    for key, value in distribution.items():
        print(f"  {key}: {value}")
    
    print("\nPreprocessing data...")
    X_train, X_test, y_train, y_test = preprocess_data(features, target)
    print(f"Training set: {X_train.shape[0]} samples")
    print(f"Test set: {X_test.shape[0]} samples")
    
    # Train all models
    all_metrics = {}
    for display_name, model_class in MODEL_CLASSES.items():
        model_name, metrics = train_and_evaluate_model(
            model_class, display_name, X_train, y_train, X_test, y_test
        )
        all_metrics[model_name] = metrics

    # Save all metrics to JSON file
    print("\n" + "=" * 50)
    print("Saving metrics to model/metrics.json...")
    print("=" * 50)
    save_metrics(all_metrics)
    print("Metrics saved successfully!")

    # Print comparison table
    print_comparison_table(all_metrics)


def print_comparison_table(all_metrics: Dict[str, Dict[str, float]]) -> None:
    print("\n" + "=" * 80)
    print("MODEL COMPARISON TABLE")
    print("=" * 80)

    metric_names = ["accuracy", "auc_score", "precision", "recall", "f1_score", "mcc_score"]
    
    model_display_names = {
        model_class().model_name: display_name 
        for display_name, model_class in MODEL_CLASSES.items()
    }

    # Print header
    header = f"{'Model':<22}"
    for metric in metric_names:
        header += f"{metric:<12}"
    print(header)
    print("-" * 94)

    # Print each model's metrics
    for model_name, metrics in all_metrics.items():
        display_name = model_display_names.get(model_name, model_name)
        row = f"{display_name:<22}"
        for metric in metric_names:
            row += f"{metrics[metric]:<12.4f}"
        print(row)

    print("-" * 94)

    # Find and print best model for each metric
    print("\nBest Model per Metric:")
    for metric in metric_names:
        best_model = max(all_metrics.keys(), key=lambda m: all_metrics[m][metric])
        best_value = all_metrics[best_model][metric]
        display_name = model_display_names.get(best_model, best_model)
        print(f"  {metric}: {display_name} ({best_value:.4f})")

    print("\n" + "=" * 80)
    print("Training complete! All models saved to model/ directory.")
    print("=" * 80)
    

if __name__ == "__main__":
    main()
