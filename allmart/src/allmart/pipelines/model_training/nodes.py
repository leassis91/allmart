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
from tqdm import tqdm
from typing import Dict, List, Tuple


def split_train(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    X = df.copy()
    X_train = X.drop(columns='customer_id', axis=1)
    y = X['monetary'].copy()
    
    return X_train, y

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
    
# def making_preds(X: pd.DataFrame, 
#                  y: pd.Series, 
#                  component_range: list, 
#                  cluster_range: list, 
#                  eps_range: list, 
#                  n_estimators=100) -> pd.DataFrame:
    
#     results = []
#     best_pipeline = None
#     best_score = -np.inf
#     best_params = None

#     scalers = {
#         "StandardScaler": StandardScaler(),
#         "MinMaxScaler": MinMaxScaler(),
#         "RobustScaler": RobustScaler(),
#     }
    
#     reductors = {
#         'PCA': PCA,  
#         'UMAP': umap.UMAP, 
#         'TreeBasedEmbedding': TreeBasedEmbedding, 
#     }
    
#     metrics = {
#         "Silhouette Score": silhouette_score,
#         "Calinski Harabasz Score": calinski_harabasz_score,
#         "Davies Bouldin Score": davies_bouldin_score,
#     }
    
#     # Main loop
#     for scaler_name, scaler in tqdm(scalers.items(), desc="Scalers"):
#         for red_name, reductor in tqdm(reductors.items()):
#             for model_name in tqdm(['DBSCAN', 'K-Means', 'Hierarchical', 'Gaussian Mixture']):
#                 if model_name == 'DBSCAN':
#                     for component in component_range:
#                         X_scaled = scaler.fit_transform(X)
                        
#                         # Properly instantiate reductor
#                         if red_name != 'TreeBasedEmbedding':
#                             X_pca = reductor(n_components=int(component), random_state=42).fit_transform(X_scaled)
#                         else:
#                             X_leaf = reductor(n_estimators=n_estimators).fit_transform(X, y)
#                             X_pca = PCA(n_components=2).fit_transform(X_leaf)
                        
#                         for eps_val in eps_range:
#                             model = DBSCAN(eps=eps_val, min_samples=20)
#                             labels = model.fit_predict(X_pca)

#                             # Calculate only if at least 5 clusters exist, removing noise
#                             unique_labels = np.unique(labels)
#                             unique_labels = unique_labels[unique_labels != -1]

#                             if len(unique_labels) >= 5:
#                                 sil = silhouette_score(X, labels)
#                                 ch = calinski_harabasz_score(X, labels)
#                                 db = davies_bouldin_score(X, labels)

#                                 # Store results
#                                 results.append({
#                                     'scaler': scaler_name,
#                                     'dim_reduction': red_name,
#                                     'n_components': component,
#                                     'model': model,
#                                     'n_clusters/eps': eps_val,
#                                     'silhouette': sil,
#                                     'calinski_harabasz': ch,
#                                     'davies_bouldin': db,
# #                                     'labels': labels
#                                 })
#                 else:
#                     for component in component_range:
#                         X_scaled = scaler.fit_transform(X)
                        
#                         # Properly instantiate reductor
#                         if red_name != 'TreeBasedEmbedding':
#                             X_pca = reductor(n_components=int(component), random_state=42).fit_transform(X_scaled)
#                         else:
#                             X_leaf = reductor(n_estimators=n_estimators).fit_transform(X, y)
#                             X_pca = PCA(n_components=2).fit_transform(X_leaf)
                        
#                         for cluster in cluster_range:
#                             if model_name == 'K-Means':
#                                 model = KMeans(n_clusters=cluster, random_state=42, n_init=20)
#                             elif model_name == 'Hierarchical':
#                                 model = AgglomerativeClustering(n_clusters=cluster)
#                             elif model_name == 'Gaussian Mixture':
#                                 model = GaussianMixture(n_components=cluster, random_state=42)
                            
#                             labels = model.fit_predict(X_pca)
#                             sil = silhouette_score(X, labels)
#                             ch = calinski_harabasz_score(X, labels)
#                             db = davies_bouldin_score(X, labels)
                            
#                             results.append({
#                                 'scaler': scaler_name,
#                                 'dim_reduction': red_name,
#                                 'n_components': component,
#                                 'model': model,
#                                 'n_clusters/eps': cluster,
#                                 'silhouette': sil,
#                                 'calinski_harabasz': ch,
#                                 'davies_bouldin': db,
# #                                 'labels': labels
#                             })
                            
#     results_no_pipeline = [{k: v for k, v in r.items() if k != 'pipeline' and k != 'labels'} for r in results]
#     results_pd = pd.DataFrame(results_no_pipeline)
#     best_result = results_pd.sort_values(['silhouette', 'calinski_harabasz', 'davies_bouldin'], ascending=[False, False, False])[0]
#     return pd.DataFrame(results_no_pipeline)


# def making_preds(X: pd.DataFrame, 
#                  y: pd.Series, 
#                  component_range: dict, 
#                  cluster_range: dict, 
#                  eps_range: dict, 
#                  n_estimators=100) -> pd.DataFrame:
    
#     # Convert YAML range structures to numpy arrays
#     component_values = np.arange(
#         component_range["start"],
#         component_range["stop"],
#         component_range.get("step", 1)  # Default step to 1 if not provided
#     )
    
#     cluster_values = np.arange(
#         cluster_range["start"],
#         cluster_range["stop"],
#         cluster_range.get("step", 1)
#     )
    
#     eps_values = np.arange(
#         eps_range["start"],
#         eps_range["stop"],
#         eps_range.get("step", 0.1)
#     )
    
#     results = []
#     best_pipeline = None
#     best_score = -np.inf
#     best_params = None

#     scalers = {
#         "StandardScaler": StandardScaler(),
#         "MinMaxScaler": MinMaxScaler(),
#         "RobustScaler": RobustScaler(),
#     }
    
#     reductors = {
#         'PCA': PCA,  
#         'UMAP': umap.UMAP, 
#         'TreeBasedEmbedding': TreeBasedEmbedding, 
#     }
    
#     metrics = {
#         "Silhouette Score": silhouette_score,
#         "Calinski Harabasz Score": calinski_harabasz_score,
#         "Davies Bouldin Score": davies_bouldin_score,
#     }
    
#     # Main loop
#     for scaler_name, scaler in tqdm(scalers.items(), desc="Scalers"):
#         for red_name, reductor in tqdm(reductors.items()):
#             for model_name in tqdm(['DBSCAN', 'K-Means', 'Hierarchical', 'Gaussian Mixture']):
#                 if model_name == 'DBSCAN':
#                     for component in component_values:  # Use the converted values
#                         X_scaled = scaler.fit_transform(X)
                        
#                         # Properly instantiate reductor
#                         if red_name != 'TreeBasedEmbedding':
#                             X_pca = reductor(n_components=int(component), random_state=42).fit_transform(X_scaled)
#                         else:
#                             X_leaf = reductor(n_estimators=n_estimators).fit_transform(X, y)
#                             X_pca = PCA(n_components=2).fit_transform(X_leaf)
                        
#                         for eps_val in eps_values:  # Use the converted values
#                             model = DBSCAN(eps=float(eps_val), min_samples=20)
#                             labels = model.fit_predict(X_pca)

#                             # Calculate only if at least 5 clusters exist, removing noise
#                             unique_labels = np.unique(labels)
#                             unique_labels = unique_labels[unique_labels != -1]

#                             if len(unique_labels) >= 5:
#                                 sil = silhouette_score(X, labels)
#                                 ch = calinski_harabasz_score(X, labels)
#                                 db = davies_bouldin_score(X, labels)

#                                 # Store results
#                                 results.append({
#                                     'scaler': scaler_name,
#                                     'dim_reduction': red_name,
#                                     'n_components': component,
#                                     'model': model,
#                                     'n_clusters/eps': eps_val,
#                                     'silhouette': sil,
#                                     'calinski_harabasz': ch,
#                                     'davies_bouldin': db,
#                                 })
#                 else:
#                     for component in component_values:  # Use the converted values
#                         X_scaled = scaler.fit_transform(X)
                        
#                         # Properly instantiate reductor
#                         if red_name != 'TreeBasedEmbedding':
#                             X_pca = reductor(n_components=int(component), random_state=42).fit_transform(X_scaled)
#                         else:
#                             X_leaf = reductor(n_estimators=n_estimators).fit_transform(X, y)
#                             X_pca = PCA(n_components=2).fit_transform(X_leaf)
                        
#                         for cluster in cluster_values:  # Use the converted values
#                             if model_name == 'K-Means':
#                                 model = KMeans(n_clusters=int(cluster), random_state=42, n_init=20)
#                             elif model_name == 'Hierarchical':
#                                 model = AgglomerativeClustering(n_clusters=int(cluster))
#                             elif model_name == 'Gaussian Mixture':
#                                 model = GaussianMixture(n_components=int(cluster), random_state=42)
                            
#                             labels = model.fit_predict(X_pca)
#                             sil = silhouette_score(X, labels)
#                             ch = calinski_harabasz_score(X, labels)
#                             db = davies_bouldin_score(X, labels)
                            
#                             results.append({
#                                 'scaler': scaler_name,
#                                 'dim_reduction': red_name,
#                                 'n_components': component,
#                                 'model': model,
#                                 'n_clusters/eps': cluster,
#                                 'silhouette': sil,
#                                 'calinski_harabasz': ch,
#                                 'davies_bouldin': db,
#                             })
                            
#     results_no_pipeline = [{k: v for k, v in r.items() if k != 'pipeline' and k != 'labels'} for r in results]
#     results_pd = pd.DataFrame(results_no_pipeline)
    
#     # Fixed this line which had an error - it was trying to get index 0 from a dataframe
#     best_result = results_pd.sort_values(['silhouette', 'calinski_harabasz', 'davies_bouldin'], 
#                                         ascending=[False, False, True]).iloc[0]
    
#     return pd.DataFrame(results_no_pipeline)


def scale_data(X: pd.DataFrame, features: list, scaler_name: str) -> pd.DataFrame:
    """Scale the data using the specified scaler."""
    X = X[features].copy()
    
    scalers = {
        "StandardScaler": StandardScaler(),
        "MinMaxScaler": MinMaxScaler(),
        "RobustScaler": RobustScaler(),
    }
    
    scaler = scalers[scaler_name]
    X_scaled = scaler.fit_transform(X)
    
    return pd.DataFrame(X_scaled, columns=features)

def reduce_dimensionality(X: pd.DataFrame, y: pd.Series = None, 
                          reductor_name: str = 'PCA', 
                          n_components: int = 2,
                          n_estimators: int = 100) -> pd.DataFrame:
    """Reduce dimensionality of the data."""
    reductors = {
        'PCA': PCA,  
        'UMAP': umap.UMAP, 
        'TreeBasedEmbedding': TreeBasedEmbedding, 
    }
    
    reductor = reductors[reductor_name]
    
    if reductor_name == 'TreeBasedEmbedding':
        X_embedded = reductor(n_estimators=n_estimators).fit_transform(X, y)
        X_reduced = PCA(n_components=2).fit_transform(X_embedded)
    else:
        X_reduced = reductor(n_components=int(n_components), random_state=42).fit_transform(X)
    
    return pd.DataFrame(X_reduced, columns=[f'component_{i}' for i in range(X_reduced.shape[1])])

def cluster_data(X: pd.DataFrame, model_type: str, param_value: float) -> dict:
    """Apply clustering algorithm and evaluate results."""
    if model_type == 'DBSCAN':
        model = DBSCAN(eps=float(param_value), min_samples=20)
    elif model_type == 'K-Means':
        model = KMeans(n_clusters=int(param_value), random_state=42, n_init=20)
    elif model_type == 'Hierarchical':
        model = AgglomerativeClustering(n_clusters=int(param_value))
    elif model_type == 'Gaussian Mixture':
        model = GaussianMixture(n_components=int(param_value), random_state=42)
    
    labels = model.fit_predict(X)
    
    # Only calculate metrics if we have more than one cluster and no noise points
    if model_type == 'DBSCAN':
        unique_labels = np.unique(labels)
        unique_labels = unique_labels[unique_labels != -1]
        if len(unique_labels) < 2:
            return None
    
    # Calculate evaluation metrics
    sil = silhouette_score(X, labels)
    ch = calinski_harabasz_score(X, labels)
    db = davies_bouldin_score(X, labels)
    
    return {
        'model_type': model_type,
        'param_value': param_value,
        'labels': labels,
        'silhouette': sil,
        'calinski_harabasz': ch,
        'davies_bouldin': db
    }

def find_best_models(scaled_data_list: List[Dict], 
                     reduced_data_list: List[Dict], 
                     model_types: List[str],
                     cluster_range: dict,
                     eps_range: dict) -> pd.DataFrame:
    """Run all combinations of scaled data, reduced data, and clustering algorithms."""
    results = []
    
    for scaled_data_dict in scaled_data_list:
        for reduced_data_dict in reduced_data_list:
            for model_type in model_types:
                param_range = eps_range if model_type == 'DBSCAN' else cluster_range
                param_values = np.arange(
                    param_range["start"],
                    param_range["stop"],
                    param_range.get("step", 0.1 if model_type == 'DBSCAN' else 1)
                )
                
                for param_value in param_values:
                    # Match scaled data with reduced data
                    if scaled_data_dict['id'] == reduced_data_dict['scaler_id']:
                        X_reduced = reduced_data_dict['data']
                        
                        cluster_result = cluster_data(X_reduced, model_type, param_value)
                        
                        if cluster_result:
                            results.append({
                                'scaler': scaled_data_dict['scaler'],
                                'dim_reduction': reduced_data_dict['reductor'],
                                'n_components': reduced_data_dict['n_components'],
                                'model': cluster_result['model_type'],
                                'n_clusters/eps': cluster_result['param_value'],
                                'silhouette': cluster_result['silhouette'],
                                'calinski_harabasz': cluster_result['calinski_harabasz'],
                                'davies_bouldin': cluster_result['davies_bouldin'],
                            })
    
    return pd.DataFrame(results)