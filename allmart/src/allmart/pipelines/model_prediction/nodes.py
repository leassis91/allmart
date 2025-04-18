"""
This is a boilerplate pipeline 'model_prediction'
generated using Kedro 0.19.11
"""

from typing import Dict, List, Union

import pandas as pd

from sklearn.cluster import DBSCAN, KMeans, AgglomerativeClustering
from sklearn.mixture import GaussianMixture



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
    
    if best_model_result is None:
        return None
    
    # Get the best model's name
    best_model_name = best_model_result['model'].__class__.__name__
    
    # Return the best model pickle
    
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