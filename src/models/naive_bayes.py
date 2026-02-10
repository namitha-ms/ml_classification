from sklearn.naive_bayes import GaussianNB
from src.models.base_model import BaseClassifier

class NaiveBayesClf(BaseClassifier):

    def __init__(self):
        super().__init__("naive_bayes")
    
    def _create_model(self):
        return GaussianNB()
