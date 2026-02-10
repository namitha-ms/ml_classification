"""
Models package for credit card default classification.

This package contains all classifier implementations that inherit from BaseClassifier.
Each model is implemented in its own module for maintainability.

Available models:
- LogisticRegressionClf: Logistic Regression classifier
- DecisionTreeClf: Decision Tree classifier
- KNNClf: K-Nearest Neighbors classifier
- NaiveBayesClf: Naive Bayes (Gaussian) classifier
- RandomForestClf: Random Forest classifier
- XGBoostClf: XGBoost classifier
"""

from src.models.base_model import BaseClassifier
from src.models.logistic_regression import LogisticRegressionClf
from src.models.decision_tree import DecisionTreeClf
from src.models.knn import KNNClf
from src.models.naive_bayes import NaiveBayesClf
from src.models.random_forest import RandomForestClf
from src.models.xgboost_model import XGBoostClf

__all__ = [
    'BaseClassifier',
    'LogisticRegressionClf',
    'DecisionTreeClf',
    'KNNClf',
    'NaiveBayesClf',
    'RandomForestClf',
    'XGBoostClf',
]
