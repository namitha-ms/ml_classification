from sklearn.ensemble import RandomForestClassifier
from src.models.base_model import BaseClassifier

class RandomForestClf(BaseClassifier):
   
    def __init__(self, n_estimators: int = 100, random_state: int = 42):
        super().__init__("random_forest")
        self.n_estimators = n_estimators
        self.random_state = random_state
    
    def _create_model(self):
        return RandomForestClassifier(
            n_estimators=self.n_estimators,
            random_state=self.random_state
        )
