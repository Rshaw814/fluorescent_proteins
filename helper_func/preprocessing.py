import numpy as np


class EmbeddingNormalizer:
    def __init__(self):
        self.mean = None
        self.std = None
        
    def fit(self, X):
        X = np.array(X)
        self.mean = np.mean(X, axis=0)
        self.std = np.std(X, axis=0) + 1e-8
        
    def transform(self, X):
        assert self.mean is not None, "Call fit() first"
        X = np.array(X)
        return (X - self.mean) / self.std
        
    def fit_transform(self, X):
        X = np.array(X)
        self.fit(X)
        
        return self.transform(X)