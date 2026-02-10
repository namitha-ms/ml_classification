from abc import ABC, abstractmethod
from typing import Dict, Any
import numpy as np

from src.model_utils import evaluate_model, save_model, load_model

class BaseClassifier(ABC):
    
    def __init__(self, model_name: str):
        self.model_name = model_name
        self.model = None
        self.metrics: Dict[str, float] = {}

    @abstractmethod
    def _create_model(self) -> Any:
        pass

    def train(self, X_train: np.ndarray, y_train: np.ndarray) -> 'BaseClassifier':
        if self.model is None:
            self.model = self._create_model()
        self.model.fit(X_train, y_train)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        if self.model is None:
            raise ValueError("Model must be trained before making predictions")
        return self.model.predict(X)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        if self.model is None:
            raise ValueError("Model must be trained before making predictions")
        return self.model.predict_proba(X)

    def evaluate(self, X_test: np.ndarray, y_test: np.ndarray) -> Dict[str, float]:
        self.metrics = evaluate_model(self.model, X_test, y_test)
        return self.metrics

    def save(self, output_dir: str = "model/") -> None:
        save_model(self.model, self.model_name, output_dir)

    def load(self, model_dir: str = "model/") -> 'BaseClassifier':
        self.model = load_model(self.model_name, model_dir)
        return self

    def get_model(self) -> Any:
        return self.model
