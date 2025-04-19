"""
This is a boilerplate pipeline 'model_prediction'
generated using Kedro 0.19.11
"""

from typing import Dict, List, Union

import pandas as pd
import numpy as np

from sklearn.cluster import DBSCAN, KMeans, AgglomerativeClustering
from sklearn.mixture import GaussianMixture



def combine_results(kmeans_results_list: List, hierarchical_results_list: List, 
                   dbscan_results_list: List, gmm_results_list: List) -> List[Dict]:
    """Combine results from different clustering methods."""
    combined_results = []
    
    # Flatten the lists and add valid models
    for results_list in [kmeans_results_list, hierarchical_results_list, 
                        dbscan_results_list, gmm_results_list]:
        if results_list:  # Check if the list exists and is not empty
            for result in results_list:
                # Check if result is a dictionary and has a model
                if isinstance(result, dict):
                    combined_results.append(result)
    
    return combined_results


def find_best_model(model_results: List[Dict]) -> Union[KMeans, AgglomerativeClustering, DBSCAN, GaussianMixture]:
    """Find the best model based on silhouette score."""
    
    # Find the model with the highest silhouette score
    best_model_result = max(model_results, key=lambda x: x['silhouette'])['model']
    
    if best_model_result is None:
        return None
    
    return best_model_result


def generate_cluster_report(X: pd.DataFrame, best_model) -> pd.DataFrame:
    """Generate a memory-efficient cluster report."""
    import gc
    
    if best_model is None:
        print("No valid model found for generating report")
        return pd.DataFrame()
    
    original_size = X.shape[0]
    
    # For very large datasets, we need to be careful with memory
    if original_size > 50000:
        sample_size = min(50000, original_size)
        print(f"Using {sample_size} samples for generating cluster report")
        sample_indices = np.random.choice(original_size, size=sample_size, replace=False)
        
        if isinstance(X, pd.DataFrame):
            X_sample = X.iloc[sample_indices].copy()
        else:
            X_sample = X[sample_indices].copy()
    else:
        X_sample = X.copy()
    
    # Generate labels based on model type
    if isinstance(best_model, AgglomerativeClustering):
        print("Handling hierarchical clustering model")
        # Handle sample size for memory constraints
        if X_sample.shape[0] > 10000:
            subsample_size = 10000
            subsample_indices = np.random.choice(X_sample.shape[0], size=subsample_size, replace=False)
            
            if isinstance(X_sample, pd.DataFrame):
                X_subsample = X_sample.iloc[subsample_indices].copy()
            else:
                X_subsample = X_sample[subsample_indices].copy()
                
            labels = best_model.fit_predict(X_subsample)
            X_with_clusters = X_subsample.copy()
        else:
            labels = best_model.fit_predict(X_sample)
            X_with_clusters = X_sample.copy()
    else:
        # For other models that have predict method
        try:
            # First try to predict with the model
            labels = best_model.predict(X_sample)
            X_with_clusters = X_sample.copy()
        except Exception as e:
            print(f"Error predicting clusters: {str(e)}")
            # For DBSCAN, we might need to fit again
            if isinstance(best_model, DBSCAN):
                print("Refitting DBSCAN model")
                labels = best_model.fit_predict(X_sample)
                X_with_clusters = X_sample.copy()
            else:
                raise e
    
    # Create a DataFrame with cluster labels
    if isinstance(X_with_clusters, pd.DataFrame):
        X_with_clusters['cluster'] = labels
    else:
        # Convert to DataFrame if it's a numpy array
        X_with_clusters = pd.DataFrame(X_with_clusters)
        X_with_clusters['cluster'] = labels
    
    # Ensure required columns exist
    required_columns = ['customer_id', 'recency', 'frequency', 'monetary', 'total_items']
    missing_columns = [col for col in required_columns if col not in X_with_clusters.columns]
    
    if missing_columns:
        print(f"Warning: Missing columns: {missing_columns}")
        # Add dummy columns for missing data
        for col in missing_columns:
            if col == 'customer_id':
                X_with_clusters[col] = range(len(X_with_clusters))
            else:
                X_with_clusters[col] = 0
    
    # Compute cluster statistics
    try:
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
        
        # Add model type information
        model_type = type(best_model).__name__
        cluster_report['model_type'] = model_type
        
        # Force garbage collection
        X_with_clusters = None
        labels = None
        gc.collect()
        
        return cluster_report
        
    except Exception as e:
        print(f"Error generating cluster report: {str(e)}")
        # Return minimal report
        return pd.DataFrame({
            'cluster': [0],
            'total_customers': [len(X_with_clusters)],
            'model_type': [type(best_model).__name__]
        })