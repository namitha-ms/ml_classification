from xgboost import XGBClassifier
from src.models.base_model import BaseClassifier

class XGBoostClf(BaseClassifier):
   
    def __init__(self, n_estimators: int = 100, random_state: int = 42):
        super().__init__("xgboost")
        self.n_estimators = n_estimators
        self.random_state = random_state
    
    def _create_model(self):
        return XGBClassifier(
            n_estimators=self.n_estimators,
            random_state=self.random_state,
            eval_metric='logloss',
            scale_pos_weight=3.5
        )
