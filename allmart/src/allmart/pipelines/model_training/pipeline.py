"""
This is a boilerplate pipeline 'model_training'
generated using Kedro 0.19.11
"""

from kedro.pipeline import node, Pipeline, pipeline  # noqa

# from .nodes import split_train, making_preds
from .nodes import (
    scale_data,
    reduce_data_with_pca,
    reduce_data_with_umap,
    apply_kmeans,
    apply_dbscan,
    apply_gmm,
    apply_hierarchical,
    combine_results,
    find_best_model,
    generate_cluster_report
)


def create_pipeline(**kwargs) -> Pipeline:
    
    # --- 1. DATA PREPROCESSING ---
    nodes = []
    # Step 1: Data Scaling
    nodes.append(
        node(
            func=scale_data,
            inputs=["Engineered_features", "RFM_features", "params:features"],
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
        node(
            func=reduce_data_with_umap,
            inputs=["Standardized_data", "params:n_components_umap", "params:random_state"],
            outputs="UMAP_Standardized_data",
            name="UMAP_SS_reduction"
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
        node(
            func=reduce_data_with_umap,
            inputs=["Minmax_data", "params:n_components_umap", "params:random_state"],
            outputs="UMAP_MinMax_data",
            name="UMAP_MM_reduction"
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
        node(
            func=reduce_data_with_umap,
            inputs=["Robust_data", "params:n_components_umap", "params:random_state"],
            outputs="UMAP_Robust_data",
            name="UMAP_RS_reduction"
        ),
    ])
    
    
 
 
    return pipeline(nodes)