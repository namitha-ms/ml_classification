from sklearn.neighbors import KNeighborsClassifier
from src.models.base_model import BaseClassifier


class KNNClf(BaseClassifier):
    def __init__(self, n_neighbors: int = 5):
        super().__init__("knn")
        self.n_neighbors = n_neighbors
    
    def _create_model(self):
        return KNeighborsClassifier(n_neighbors=self.n_neighbors)
