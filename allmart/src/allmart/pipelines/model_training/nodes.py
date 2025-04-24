"""
This is a boilerplate pipeline 'model_training'
generated using Kedro 0.19.11
"""

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from sklearn.decomposition import PCA
from sklearn.cluster import DBSCAN, KMeans, AgglomerativeClustering, MiniBatchKMeans
from sklearn.mixture import GaussianMixture
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
import pandas as pd
import numpy as np
import umap.umap_ as umap
# from tqdm import tqdm
from typing import Dict, List, Union, Tuple
import gc


# Step 1: Scaling
def scale_data(df_engineered: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Scale the data using the specified scaler."""
    X = df_engineered.copy()
    
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

# Step 3: Clustering
def apply_kmeans(X_reduced: pd.DataFrame, cluster_range: dict) -> Dict:
    """Apply memory-efficient K-Means clustering."""
    
    # For very large datasets, use sampling for evaluation only (we'll train on full data)
    original_size = X_reduced.shape[0]
    evaluation_set = X_reduced
    
    if original_size > 20000:
        sample_size = 20000
        print(f"KMeans: Using {sample_size} samples for evaluation from {original_size}")
        sample_indices = np.random.choice(original_size, size=sample_size, replace=False)
        
        # Check if X_reduced is a DataFrame or ndarray and handle accordingly
        if isinstance(X_reduced, pd.DataFrame):
            evaluation_set = X_reduced.iloc[sample_indices].copy()
        else:
            evaluation_set = X_reduced[sample_indices].copy()
    
    cluster_values = range(
        cluster_range["start"],
        cluster_range["stop"],
        cluster_range.get("step", 1)
    )
    
    best_score = -1
    best_n_clusters = None
    
    for n_clusters in cluster_values:
        # Use more efficient settings for large datasets
        if original_size > 100000:
            # Use mini-batch for very large datasets
            model = MiniBatchKMeans(
                n_clusters=int(n_clusters), 
                random_state=42, 
                batch_size=1000,
                max_iter=100,
                n_init=3
            )
        else:
            # Use regular KMeans with optimized settings
            model = KMeans(
                n_clusters=int(n_clusters), 
                random_state=42, 
                n_init=10 if original_size < 50000 else 3,
                max_iter=300,
                algorithm='elkan'  # Usually faster for low-dimensional data
            )
        
        try:
            # Fit on full dataset or subsampled data for very large datasets
            if original_size > 500000:
                training_sample_size = 100000
                training_indices = np.random.choice(original_size, size=training_sample_size, replace=False)
                
                # Handle different input types
                if isinstance(X_reduced, pd.DataFrame):
                    training_data = X_reduced.iloc[training_indices]
                else:
                    training_data = X_reduced[training_indices]
                
                model.fit(training_data)
            else:
                model.fit(X_reduced)
                
            # Predict on evaluation set
            labels = model.predict(evaluation_set)
            
            # Check if we have enough samples in each cluster for silhouette
            unique_labels, counts = np.unique(labels, return_counts=True)
            if len(unique_labels) < 2 or np.any(counts < 2):
                print(f"KMeans n_clusters={n_clusters}: Not enough samples per cluster")
                continue
            
            sil = silhouette_score(evaluation_set, labels)
            print(f"KMeans n_clusters={n_clusters}: silhouette={sil:.4f}")
            
            if sil > best_score:
                best_score = sil
                best_n_clusters = n_clusters
                print(f"  New best KMeans model: n_clusters={n_clusters}, silhouette={sil:.4f}")
        except Exception as e:
            print(f"Error in KMeans with n_clusters={n_clusters}: {str(e)}")
            continue
        
        # Clean up memory
        model = None
        labels = None
        gc.collect()
    
    if best_n_clusters is None:
        print("KMeans: No valid models found")
        return {
            'model': None,
            'silhouette': -1,
            'n_clusters': 0
        }
    
    # Create final model with optimal parameters
    # Always use regular KMeans for the final model to ensure proper functionality
    final_model = KMeans(
        n_clusters=int(best_n_clusters), 
        random_state=42, 
        n_init=10,
        algorithm='elkan'
    )
    
    print(f"Final KMeans model: n_clusters={best_n_clusters}, silhouette={best_score:.4f}")
    
    return {
        'model': final_model,
        'silhouette': best_score,
        'n_clusters': best_n_clusters
    }

def apply_hierarchical(X_reduced: pd.DataFrame, cluster_range: dict) -> Dict:
    """Apply memory-efficient hierarchical clustering."""
    # If dataset is too large, subsample it
    if X_reduced.shape[0] > 10000:
        # Randomly sample 10,000 points for hierarchical clustering
        sample_size = 10000
        sample_indices = np.random.choice(X_reduced.shape[0], size=sample_size, replace=False)
        X_sample = X_reduced[sample_indices]
    else:
        X_sample = X_reduced
        sample_indices = None
    
    cluster_values = range(
        cluster_range["start"],
        cluster_range["stop"],
        cluster_range.get("step", 1)
    )
    
    best_score = -1
    best_model = None
    best_n_clusters = None
    
    # Only iterate over a smaller range if dataset is very large
    if X_reduced.shape[0] > 50000:
        effective_range = range(cluster_range["start"], min(8, cluster_range["stop"]), 1)
    else:
        effective_range = cluster_values
    
    for n_clusters in effective_range:
        # Use memory-efficient settings
        model = AgglomerativeClustering(
            n_clusters=int(n_clusters),
            distance_threshold=None,
            compute_full_tree=False,
            linkage='ward'  # Ward linkage is more memory-efficient
        )
        labels = model.fit_predict(X_sample)
        
        try:
            sil = silhouette_score(X_sample, labels)
            if sil > best_score:
                best_score = sil
                best_model = model
                best_n_clusters = n_clusters
        except:
            continue
        
        # Force garbage collection
        gc.collect()
    
    if best_model is None:
        return {
            'model': None,
            'silhouette': -1,
            'n_clusters': 0
        }
    
    # Create a fresh model with the best parameters
    final_model = AgglomerativeClustering(n_clusters=int(best_n_clusters), linkage='ward')
    
    return {
        'model': final_model,
        'silhouette': best_score,
        'n_clusters': best_n_clusters
    }


def apply_dbscan(X_reduced: pd.DataFrame, eps_range: dict, min_samples: int = 5) -> Dict:
    """Apply DBSCAN clustering with optimized memory usage."""
    eps_values = np.arange(
        eps_range["start"],
        eps_range["stop"],
        eps_range.get("step", 0.1)
    )
    
    best_score = -1
    best_model = None
    best_eps = None
    
    # Process in smaller batches to reduce memory usage
    batch_size = 5  # Process 5 eps values at a time
    
    for i in range(0, len(eps_values), batch_size):
        batch_eps = eps_values[i:i+batch_size]
        
        for eps_val in batch_eps:
            # Use a fresh model each time to prevent memory accumulation
            model = DBSCAN(eps=float(eps_val), min_samples=min_samples)
            labels = model.fit_predict(X_reduced)
            
            # Free up memory
            gc.collect()
            
            # Check if we have at least 2 clusters (excluding noise)
            unique_labels = np.unique(labels)
            unique_labels = unique_labels[unique_labels != -1]
            
            if len(unique_labels) >= 2:
                try:
                    # Calculate silhouette score with smaller sample if dataset is large
                    if X_reduced.shape[0] > 10000:
                        sample_indices = np.random.choice(X_reduced.shape[0], size=10000, replace=False)
                        sil = silhouette_score(X_reduced[sample_indices], labels[sample_indices])
                    else:
                        sil = silhouette_score(X_reduced, labels)
                        
                    if sil > best_score:
                        best_score = sil
                        # Store only parameters not the model itself to save memory
                        best_eps = eps_val
                except:
                    continue
        
        # Force garbage collection between batches
        gc.collect()
    
    # Only create the final model once we know the best parameters
    if best_eps is not None:
        best_model = DBSCAN(eps=float(best_eps), min_samples=min_samples)
    else:
        return {
            'model': None,
            'silhouette': -1,
            'n_clusters': 0
        }
    
    labels = best_model.fit_predict(X_reduced)
    unique_labels = np.unique(labels)
    unique_labels = unique_labels[unique_labels != -1]
    
    return {
        'model': best_model,
        'silhouette': best_score,
        'n_clusters': len(unique_labels)
    }

def apply_gmm(X_reduced: pd.DataFrame, cluster_range: dict) -> Dict:
    """Apply memory-efficient Gaussian Mixture Model clustering."""
    import gc
    
    # For very large datasets, use sampling for evaluation
    original_size = X_reduced.shape[0]
    evaluation_set = X_reduced
    
    if original_size > 20000:
        sample_size = 20000
        print(f"GMM: Using {sample_size} samples for evaluation from {original_size}")
        sample_indices = np.random.choice(original_size, size=sample_size, replace=False)
        evaluation_set = X_reduced[sample_indices].copy()
    
    # For very large datasets, train on subset
    training_set = X_reduced
    if original_size > 100000:
        training_size = 50000
        print(f"GMM: Using {training_size} samples for training from {original_size}")
        training_indices = np.random.choice(original_size, size=training_size, replace=False)
        training_set = X_reduced[training_indices].copy()
    
    cluster_values = range(
        cluster_range["start"],
        cluster_range["stop"],
        cluster_range.get("step", 1)
    )
    
    # For large datasets, limit the range
    if original_size > 50000:
        max_clusters = min(8, cluster_range["stop"])
        effective_range = range(cluster_range["start"], max_clusters, cluster_range.get("step", 1))
        print(f"GMM: Limited cluster range to {list(effective_range)}")
    else:
        effective_range = cluster_values
    
    best_score = -1
    best_n_clusters = None
    
    for n_clusters in effective_range:
        # Use optimized settings for GMM
        model = GaussianMixture(
            n_components=int(n_clusters),
            random_state=42,
            covariance_type='full',  # Default, but explicit
            max_iter=100,            # Reduced from default 100
            n_init=1 if original_size > 50000 else 3  # Fewer initializations for large datasets
        )
        
        try:
            # Fit on training set
            model.fit(training_set)
            
            # Predict on evaluation set
            labels = model.predict(evaluation_set)
            
            # Check if we have enough samples in each cluster for silhouette
            unique_labels, counts = np.unique(labels, return_counts=True)
            if len(unique_labels) < 2 or np.any(counts < 2):
                print(f"GMM n_components={n_clusters}: Not enough samples per cluster")
                continue
            
            sil = silhouette_score(evaluation_set, labels)
            print(f"GMM n_components={n_clusters}: silhouette={sil:.4f}")
            
            if sil > best_score:
                best_score = sil
                best_n_clusters = n_clusters
                print(f"  New best GMM model: n_components={n_clusters}, silhouette={sil:.4f}")
        except Exception as e:
            print(f"Error in GMM with n_components={n_clusters}: {str(e)}")
            continue
        
        # Clean up memory
        model = None
        labels = None
        gc.collect()
    
    if best_n_clusters is None:
        print("GMM: No valid models found")
        return {
            'model': None,
            'silhouette': -1,
            'n_clusters': 0
        }
    
    # Create final model with optimal parameters
    final_model = GaussianMixture(
        n_components=int(best_n_clusters),
        random_state=42,
        covariance_type='full',
        n_init=3  # More initializations for final model
    )
    
    print(f"Final GMM model: n_components={best_n_clusters}, silhouette={best_score:.4f}")
    
    return {
        'model': final_model,
        'silhouette': best_score,
        'n_clusters': best_n_clusters
    }