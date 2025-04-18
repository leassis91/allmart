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
import umap.umap_ as umap
# from tqdm import tqdm
from typing import Dict, List, Union, Tuple

# Step 1: Scaling
def scale_data(df_engineered: pd.DataFrame, df_rfm: pd.DataFrame, features: list) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Scale the data using the specified scaler."""
    X = pd.merge(df_engineered, df_rfm, on='customer_id', how='left')
    X = X[features].copy()
    
    
    ss = StandardScaler() # Normalization
    mm = MinMaxScaler() # Scaling
    rs = RobustScaler() 
    
    X_normalized = ss.fit_transform(X)
    X_minmax =  mm.fit_transform(X)
    X_robust = rs.fit_transform(X)
    
    return X_normalized, X_minmax, X_robust

# Step 2: Dimensionality Reduction
def reduce_data_with_pca(X_scaled: pd.DataFrame, n_components_pca: int, random_state: int) -> pd.DataFrame:
    """
    Apply PCA with different component counts.
    """
    
    n_components = n_components_pca
    pca = PCA(n_components=int(n_components), random_state=random_state)
    X_reduced = pca.fit_transform(X_scaled)
    
    return X_reduced

def reduce_data_with_umap(X: pd.DataFrame, n_components_umap: int, random_state: int) -> pd.DataFrame:
    """
    Apply UMAP with different component counts.
    """
    
    n_components = n_components_umap
    reducer = umap.UMAP(n_components=int(n_components), random_state=random_state)
    X_reduced = reducer.fit_transform(X)
    
    return X_reduced

# Step 3: Clustering
def apply_kmeans(X_reduced: pd.DataFrame, cluster_range: Dict[str, int], n_init: int) -> dict:
    """Apply K-Means clustering to each dimensionality reduction result."""
    
    cluster_values = range(
        cluster_range["start"],
        cluster_range["stop"],
        cluster_range.get("step", 1)
    )
    
    best_score = -np.inf
    best_model = None
    best_n_clusters = None        
    
    
    for n_clusters in cluster_values:
        model = KMeans(n_clusters=int(n_clusters), n_init=n_init)
        labels = model.fit_predict(X_reduced)
        
        # Calculate metrics
        # sil = silhouette_score(X_reduced, labels)
        # ch = calinski_harabasz_score(X_reduced, labels)
        # db = davies_bouldin_score(X_reduced, labels)
        
        try:
            sil = silhouette_score(X_reduced, labels)
            if sil > best_score:
                best_score = sil
                best_model = model
                best_n_clusters = n_clusters
        except:
            continue
        
    results = {
        'model': best_model,
        'silhouette_score': best_score,
        'n_clusters': best_n_clusters,
    }
    
    return results

def apply_hierarchical(X_reduced: pd.DataFrame, cluster_range: Dict[str, int]) -> dict:
    """Apply Hierarchical clustering to each dimensionality reduction result."""
    
    cluster_values = range(
        cluster_range["start"],
        cluster_range["stop"],
        cluster_range.get("step", 1)
    )
    
    best_score = -np.inf
    best_model = None
    best_n_clusters = None
    
    for n_clusters in cluster_values:
        model = AgglomerativeClustering(n_clusters=int(n_clusters))
        labels = model.fit_predict(X_reduced)
        
        try:
            sil = silhouette_score(X_reduced, labels)
            if sil > best_score:
                best_score = sil
                best_model = model
                best_n_clusters = n_clusters
        except:
            continue
        
    results = {
        'model': best_model,
        'silhouette_score': best_score,
        'n_clusters': best_n_clusters,
    }
    
    return results


def apply_dbscan(X_reduced: pd.DataFrame, eps_range: Dict[str, int], min_samples: int) -> dict:
    """Apply DBSCAN clustering to each dimensionality reduction result."""
    results = []
    
    eps_values = np.arange(
        eps_range["start"],
        eps_range["stop"],
        eps_range.get("step", 0.1)
    )
    
    best_score = -np.inf
    best_model = None
    best_eps = None
    
    for eps_val in eps_values:
        model = DBSCAN(eps=float(eps_val), min_samples=min_samples)
        labels = model.fit_predict(X_reduced)
        
        # Check if we have meaningful clusters
        unique_labels = np.unique(labels)
        unique_labels = unique_labels[unique_labels != -1]
        
        if len(unique_labels) >= 2:
            try:
                sil = silhouette_score(X_reduced, labels)
                if sil > best_score:
                    best_score = sil
                    best_model = model
                    best_eps = eps_val
            except:
                continue
            
    
    results = {
        'model': best_model,
        'silhouette': best_score,
        'n_clusters': len(unique_labels),
        'n_eps': best_eps
    }
    
    return results

def apply_gmm(X_reduced: pd.DataFrame, cluster_range: Dict[str, int]) -> dict:
    """Apply Gaussian Mixture Model clustering to each dimensionality reduction result."""
    
    cluster_values = range(
        cluster_range["start"],
        cluster_range["stop"],
        cluster_range.get("step", 1)
    )
    
    best_score = -1
    best_model = None
    best_n_clusters = None
    
    for n_clusters in cluster_values:
        model = GaussianMixture(n_components=int(n_clusters))
        labels = model.fit_predict(X_reduced)
        
        try:
            sil = silhouette_score(X_reduced, labels)
            if sil > best_score:
                best_score = sil
                best_model = model
                best_n_clusters = n_clusters
        except:
            continue
        
    results = {
        'model': best_model,
        'silhouette': best_score,
        'n_clusters': best_n_clusters,
    }

    return results