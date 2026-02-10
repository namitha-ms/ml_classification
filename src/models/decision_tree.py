from sklearn.tree import DecisionTreeClassifier
from src.models.base_model import BaseClassifier

class DecisionTreeClf(BaseClassifier):
   
    def __init__(self, random_state: int = 42):
        super().__init__("decision_tree")
        self.random_state = random_state
    
    def _create_model(self):
        return DecisionTreeClassifier(
            class_weight='balanced',
            max_depth=8,
            min_samples_leaf=50,
            random_state=self.random_state
        )
