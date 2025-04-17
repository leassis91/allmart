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


def scale_data(df_engineered: pd.DataFrame, df_rfm: pd.DataFrame, features: list) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Scale the data using the specified scaler."""
    X = pd.merge(df_engineered, df_rfm, on='customer_id', how='left')
    X = X[features].copy()
    
    
    ss = StandardScaler()
    mm = MinMaxScaler()
    rs = RobustScaler()
    
    X_standardized = ss.fit_transform(X)
    X_normalized =  mm.fit_transform(X)
    X_robusted = rs.fit_transform(X)
    
    return X_standardized, X_normalized, X_robusted


def reduce_data_with_pca(X: pd.DataFrame, component_lenght: int, random_state: int) -> pd.DataFrame:
    """
    Apply PCA with different component counts.
    """
    
    n_components = component_lenght
    pca = PCA(n_components=int(n_components), random_state=random_state)
    X_reduced = pca.fit_transform(X)
    
    
    return X_reduced

def reduce_data_with_umap(X: pd.DataFrame, component_length: int, random_state: int) -> pd.DataFrame:
    """
    Apply UMAP with different component counts.
    """
    
    n_components = component_length
    reducer = umap.UMAP(n_components=int(n_components), random_state=random_state)
    X_reduced = reducer.fit_transform(X)
    
    return X_reduced



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


def combine_results(kmeans_results: Dict, hierarchical_results: Dict, dbscan_results: Dict, gmm_results: Dict) -> List[Dict]:
    """Combine results from different clustering methods."""
    combined_results = []
    
    # Combine results into a list
    combined_results.append(kmeans_results)
    combined_results.append(hierarchical_results)
    combined_results.append(dbscan_results)
    combined_results.append(gmm_results)
    
    # Filter out None models
    combined_results = [result for result in combined_results if result['model'] is not None]
    
    return combined_results


def find_best_model(model_results: List[Dict]) -> Union[KMeans, AgglomerativeClustering, DBSCAN, GaussianMixture]:
    """Find the best model based on silhouette score."""
    
    # Find the model with the highest silhouette score
    best_model_result = max(model_results, key=lambda x: x['silhouette'])['model']
    return best_model_result


def generate_cluster_report(X: pd.DataFrame, best_model_result: Union[KMeans, AgglomerativeClustering, DBSCAN, GaussianMixture]) -> pd.DataFrame:
    """Generate a report of cluster statistics."""
    if best_model_result['model'] is None:
        return pd.DataFrame()
    
    model = best_model_result
    
    # Generate labels
    labels = model.fit_predict(X)
    
    # Create a DataFrame with original data and cluster labels
    X_with_clusters = X.copy()
    X_with_clusters['cluster'] = labels
    
    # Compute cluster statistics
    cluster_report = X_with_clusters.groupby('cluster').agg({
        'customer_id': 'count',    
        'recency': 'mean',         
        'frequency': 'mean',
        'monetary': 'mean',
        'total_items': 'mean'
    }).reset_index()
    
    # Rename columns for clarity
    cluster_report.rename(columns={
        'customer_id': 'total_customers',
        'recency': 'avg_recency',
        'frequency': 'avg_frequency',
        'monetary': 'avg_monetary',
        'total_items': 'avg_total_items'
    }, inplace=True)
    
    
    return cluster_report