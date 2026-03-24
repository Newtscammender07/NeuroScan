import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

class PCAOptimizer:
    def __init__(self, n_components=0.95, random_state=42):
        """
        Initializes the PCA Optimizer.
        n_components: float (variance to keep) or int (number of components).
        """
        self.scaler = StandardScaler()
        self.pca = PCA(n_components=n_components, random_state=random_state)
        
    def fit_transform(self, X):
        """
        Fits the scaler and PCA on the training data, then transforms it.
        X shape: (n_samples, n_features)
        Returns:
            X_pca: Transformed data
        """
        # 1. Scale data
        X_scaled = self.scaler.fit_transform(X)
        # 2. Apply PCA
        X_pca = self.pca.fit_transform(X_scaled)
        
        return X_pca
        
    def transform(self, X):
        """
        Transforms testing/new data using the fitted scaler and PCA.
        """
        X_scaled = self.scaler.transform(X)
        return self.pca.transform(X_scaled)
        
    def get_explained_variance(self):
        """
        Returns the explained variance ratio of the components.
        """
        return self.pca.explained_variance_ratio_

    def get_cumulative_variance(self):
        """
        Returns the cumulative explained variance.
        """
        return np.cumsum(self.pca.explained_variance_ratio_)
