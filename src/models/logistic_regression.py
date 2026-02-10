from sklearn.linear_model import LogisticRegression
from src.models.base_model import BaseClassifier


class LogisticRegressionClf(BaseClassifier):
   
    def __init__(self, max_iter: int = 1000, random_state: int = 42):
        super().__init__("logistic_regression")
        self.max_iter = max_iter
        self.random_state = random_state
    
    def _create_model(self):
        return LogisticRegression(
            max_iter=self.max_iter,
            random_state=self.random_state,
            class_weight='balanced'
        )
