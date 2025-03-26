"""
This is a boilerplate pipeline 'model_training'
generated using Kedro 0.19.11
"""

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestRegressor
from sklearn.cluster import DBSCAN, KMeans, AgglomerativeClustering
from sklearn.mixture import GaussianMixture
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
import pandas as pd
import numpy as np
import umap
from tqdm import tqdm


def split_train(df):
    X = df.copy()
    X_train = X.drop(columns='customer_id', axis=1)    
    
    return X_train


class TreeBasedEmbedding(BaseEstimator, TransformerMixin):
    """
    Custom transformer that uses Random Forest's apply method to create tree-based embeddings.
    This always will only use 2 components.
    """
    def __init__(self, n_components=2, n_estimators=100, random_state=42):
        self.n_components = 2  # Hardcoded to 2 as per your original implementation
        self.n_estimators = n_estimators
        self.random_state = random_state
        self.rf = None
        
    def fit(self, X, y=None):
        # Validate y
        if y is None:
            raise ValueError("A target variable is required for fitting")
        
        # Ensure y is a Series or can be converted to one
        if not isinstance(y, pd.Series):
            y = pd.Series(y)
        
        # Handle DataFrame or numpy array input
        if isinstance(X, pd.DataFrame):
            X_features = X.drop(columns=[y.name])
        else:
            X_features = X
        
        # Train a RandomForestRegressor
        self.rf = RandomForestRegressor(
            n_estimators=self.n_estimators,
            random_state=self.random_state
        )
        self.rf.fit(X_features, y)
        return self
    
    def transform(self, X):
        # Get leaf indices for each sample across all trees
        if self.rf is None:
            raise ValueError("Transformer not fitted. Call fit before transform.")
        
        # Handle DataFrame or numpy array input
        if isinstance(X, pd.DataFrame):
            X_features = X.values
        else:
            X_features = X
        
        leaf_indices = self.rf.apply(X_features)
        return leaf_indices
    
    def fit_transform(self, X, y):
        # Validate y
        if not isinstance(y, pd.Series):
            y = pd.Series(y)
        
        # Handle DataFrame or numpy array input
        if isinstance(X, pd.DataFrame):
            X_features = X.drop(columns=[y.name])
        else:
            X_features = X
        
        # Train a RandomForestRegressor
        self.rf = RandomForestRegressor(
            n_estimators=self.n_estimators,
            random_state=self.random_state
        )
        self.rf.fit(X_features, y)
        
        leaf_indices = self.rf.apply(X_features)
        return leaf_indices
    


def making_preds(X, y, component_range, cluster_range, eps_range, n_estimators=100):
    results = []
    best_pipeline = None
    best_score = -np.inf
    best_params = None

    scalers = {
        "StandardScaler": StandardScaler(),
        "MinMaxScaler": MinMaxScaler(),
        "RobustScaler": RobustScaler(),
    }
    
    reductors = {
        'PCA': PCA,  
        'UMAP': umap.UMAP, 
        'TreeBasedEmbedding': TreeBasedEmbedding, 
    }
    
    metrics = {
        "Silhouette Score": silhouette_score,
        "Calinski Harabasz Score": calinski_harabasz_score,
        "Davies Bouldin Score": davies_bouldin_score,
    }
    
    # Main loop
    for scaler_name, scaler in tqdm(scalers.items(), desc="Scalers"):
        for red_name, reductor in tqdm(reductors.items()):
            for model_name in tqdm(['DBSCAN', 'K-Means', 'Hierarchical', 'Gaussian Mixture']):
                if model_name == 'DBSCAN':
                    for component in component_range:
                        X_scaled = scaler.fit_transform(X)
                        
                        # Properly instantiate reductor
                        if red_name != 'TreeBasedEmbedding':
                            X_pca = reductor(n_components=component, random_state=42).fit_transform(X_scaled)
                        else:
                            X_leaf = reductor(n_estimators=n_estimators).fit_transform(X, y)
                            X_pca = PCA(n_components=2).fit_transform(X_leaf)
                        
                        for eps_val in eps_range:
                            model = DBSCAN(eps=eps_val, min_samples=20)
                            labels = model.fit_predict(X_pca)

                            # Calculate only if at least 5 clusters exist, removing noise
                            unique_labels = np.unique(labels)
                            unique_labels = unique_labels[unique_labels != -1]

                            if len(unique_labels) >= 5:
                                sil = silhouette_score(X, labels)
                                ch = calinski_harabasz_score(X, labels)
                                db = davies_bouldin_score(X, labels)

                                # Store results
                                results.append({
                                    'scaler': scaler_name,
                                    'dim_reduction': red_name,
                                    'n_components': component,
                                    'model': model,
                                    'n_clusters/eps': eps_val,
                                    'silhouette': sil,
                                    'calinski_harabasz': ch,
                                    'davies_bouldin': db,
#                                     'labels': labels
                                })
                else:
                    for component in component_range:
                        X_scaled = scaler.fit_transform(X)
                        
                        # Properly instantiate reductor
                        if red_name != 'TreeBasedEmbedding':
                            X_pca = reductor(n_components=component, random_state=42).fit_transform(X_scaled)
                        else:
                            X_leaf = reductor(n_estimators=n_estimators).fit_transform(X, y)
                            X_pca = PCA(n_components=2).fit_transform(X_leaf)
                        
                        for cluster in cluster_range:
                            if model_name == 'K-Means':
                                model = KMeans(n_clusters=cluster, random_state=42, n_init=20)
                            elif model_name == 'Hierarchical':
                                model = AgglomerativeClustering(n_clusters=cluster)
                            elif model_name == 'Gaussian Mixture':
                                model = GaussianMixture(n_components=cluster, random_state=42)
                            
                            labels = model.fit_predict(X_pca)
                            sil = silhouette_score(X, labels)
                            ch = calinski_harabasz_score(X, labels)
                            db = davies_bouldin_score(X, labels)
                            
                            results.append({
                                'scaler': scaler_name,
                                'dim_reduction': red_name,
                                'n_components': component,
                                'model': model,
                                'n_clusters/eps': cluster,
                                'silhouette': sil,
                                'calinski_harabasz': ch,
                                'davies_bouldin': db,
#                                 'labels': labels
                            })
                            
    results_no_pipeline = [{k: v for k, v in r.items() if k != 'pipeline' and k != 'labels'} for r in results]
    results_pd = pd.DataFrame(results_no_pipeline)
    best_result = results_pd.sort_values(['silhouette', 'calinski_harabasz', 'davies_bouldin'], ascending=[False, False, False])[0]
    return pd.DataFrame(results_no_pipeline)