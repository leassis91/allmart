"""
This is a boilerplate pipeline 'model_training'
generated using Kedro 0.19.11
"""

from kedro.pipeline import node, Pipeline, pipeline  # noqa

# from .nodes import split_train, making_preds
from .nodes import (
    scale_data,
    reduce_data_with_pca,
    # reduce_data_with_umap,
    apply_kmeans,
    apply_dbscan,
    apply_gmm,
    apply_hierarchical,
)


def create_pipeline(**kwargs) -> Pipeline:
    
    # --- 1. DATA PREPROCESSING ---
    nodes = []
    # Step 1: Data Scaling
    nodes.append(
        node(
            func=scale_data,
            inputs="RFM_features",
            outputs=["Standardized_data", "Minmax_data", "Robust_data"],
            name="data_scaling"
        )
    )
    
     # --- 2. DIMENSIONALITY REDUCTION NODES ---
    # Standard Scaler reductions
    nodes.extend([
        node(
            func=reduce_data_with_pca,
            inputs=["Standardized_data", "params:n_components_pca", "params:random_state"],
            outputs="PCA_Standardized_data",
            name="PCA_SS_reduction"
        ),
    ])
    
    # MinMax Scaler reductions
    nodes.extend([
        node(
            func=reduce_data_with_pca,
            inputs=["Minmax_data", "params:n_components_pca", "params:random_state"],
            outputs="PCA_MinMax_data",
            name="PCA_MM_reduction"
        ),
    ])
    
    # Robust Scaler reductions
    nodes.extend([
        node(
            func=reduce_data_with_pca,
            inputs=["Robust_data", "params:n_components_pca", "params:random_state"],
            outputs="PCA_Robust_data",
            name="PCA_RS_reduction"
        ),
    ])
    
    
     # --- 3. CLUSTERING NODES ---
    # Standard + PCA clustering models
    nodes.extend([
        node(
            func=apply_kmeans,
            inputs=["PCA_Standardized_data", "params:cluster_range"],
            outputs="standard_pca_kmeans_results",
            name="standard_pca_kmeans"
        ),
        node(
            func=apply_hierarchical,
            inputs=["PCA_Standardized_data", "params:cluster_range"],
            outputs="standard_pca_hierarchical_results",
            name="standard_pca_hierarchical"
        ),
        node(
            func=apply_dbscan,
            inputs=["PCA_Standardized_data", "params:eps_range", "params:min_samples"],
            outputs="standard_pca_dbscan_results",
            name="standard_pca_dbscan"
        ),
        node(
            func=apply_gmm,
            inputs=["PCA_Standardized_data", "params:cluster_range"],
            outputs="standard_pca_gmm_results",
            name="standard_pca_gmm"
        ),
    ])
    
    # MinMax + PCA clustering models
    nodes.extend([
        node(
            func=apply_kmeans,
            inputs=["PCA_MinMax_data", "params:cluster_range"],
            outputs="minmax_pca_kmeans_results",
            name="minmax_pca_kmeans"
        ),
        node(
            func=apply_hierarchical,
            inputs=["PCA_MinMax_data", "params:cluster_range"],
            outputs="minmax_pca_hierarchical_results",
            name="minmax_pca_hierarchical"
        ),
        node(
            func=apply_dbscan,
            inputs=["PCA_MinMax_data", "params:eps_range", "params:min_samples"],
            outputs="minmax_pca_dbscan_results",
            name="minmax_pca_dbscan"
        ),
        node(
            func=apply_gmm,
            inputs=["PCA_MinMax_data", "params:cluster_range"],
            outputs="minmax_pca_gmm_results",
            name="minmax_pca_gmm"
        ),
    ])
    
    # Robust + PCA clustering models
    nodes.extend([
        node(
            func=apply_kmeans,
            inputs=["PCA_Robust_data", "params:cluster_range"],
            outputs="robust_pca_kmeans_results",
            name="robust_pca_kmeans"
        ),
        node(
            func=apply_hierarchical,
            inputs=["PCA_Robust_data", "params:cluster_range"],
            outputs="robust_pca_hierarchical_results",
            name="robust_pca_hierarchical"
        ),
        node(
            func=apply_dbscan,
            inputs=["PCA_Robust_data", "params:eps_range", "params:min_samples"],
            outputs="robust_pca_dbscan_results",
            name="robust_pca_dbscan"
        ),
        node(
            func=apply_gmm,
            inputs=["PCA_Robust_data", "params:cluster_range"],
            outputs="robust_pca_gmm_results",
            name="robust_pca_gmm"
        ),
    ])
    
    return pipeline(nodes)