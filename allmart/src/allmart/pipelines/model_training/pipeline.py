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
    
    
     # --- 3. CLUSTERING NODES ---
    # Standard + PCA clustering models
    nodes.extend([
        node(
            func=apply_kmeans,
            inputs=["PCA_Standardized_data", "params:cluster_range", "params:n_init"],
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
    
    # Standard + UMAP clustering models
    nodes.extend([
        node(
            func=apply_kmeans,
            inputs=["UMAP_Standardized_data", "params:cluster_range", "params:n_init"],
            outputs="standard_umap_kmeans_results",
            name="standard_umap_kmeans"
        ),
        node(
            func=apply_hierarchical,
            inputs=["UMAP_Standardized_data", "params:cluster_range"],
            outputs="standard_umap_hierarchical_results",
            name="standard_umap_hierarchical"
        ),
        node(
            func=apply_dbscan,
            inputs=["UMAP_Standardized_data", "params:eps_range", "params:min_samples"],
            outputs="standard_umap_dbscan_results",
            name="standard_umap_dbscan"
        ),
        node(
            func=apply_gmm,
            inputs=["UMAP_Standardized_data", "params:cluster_range"],
            outputs="standard_umap_gmm_results",
            name="standard_umap_gmm"
        ),
    ])
    
    # MinMax + PCA clustering models
    nodes.extend([
        node(
            func=apply_kmeans,
            inputs=["PCA_MinMax_data", "params:cluster_range", "params:n_init"],
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
    
    # MinMax + UMAP clustering models
    nodes.extend([
        node(
            func=apply_kmeans,
            inputs=["UMAP_MinMax_data", "params:cluster_range", "params:n_init"],
            outputs="minmax_umap_kmeans_results",
            name="minmax_umap_kmeans"
        ),
        node(
            func=apply_hierarchical,
            inputs=["UMAP_MinMax_data", "params:cluster_range"],
            outputs="minmax_umap_hierarchical_results",
            name="minmax_umap_hierarchical"
        ),
        node(
            func=apply_dbscan,
            inputs=["UMAP_MinMax_data", "params:eps_range", "params:min_samples"],
            outputs="minmax_umap_dbscan_results",
            name="minmax_umap_dbscan"
        ),
        node(
            func=apply_gmm,
            inputs=["UMAP_MinMax_data", "params:cluster_range"],
            outputs="minmax_umap_gmm_results",
            name="minmax_umap_gmm"
        ),
    ])
    
    # Robust + PCA clustering models
    nodes.extend([
        node(
            func=apply_kmeans,
            inputs=["PCA_Robust_data", "params:cluster_range", "params:n_init"],
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
    
    # Robust + UMAP clustering models
    nodes.extend([
        node(
            func=apply_kmeans,
            inputs=["UMAP_Robust_data", "params:cluster_range", "params:n_init"],
            outputs="robust_umap_kmeans_results",
            name="robust_umap_kmeans"
        ),
        node(
            func=apply_hierarchical,
            inputs=["UMAP_Robust_data", "params:cluster_range"],
            outputs="robust_umap_hierarchical_results",
            name="robust_umap_hierarchical"
        ),
        node(
            func=apply_dbscan,
            inputs=["UMAP_Robust_data", "params:eps_range", "params:min_samples"],
            outputs="robust_umap_dbscan_results",
            name="robust_umap_dbscan"
        ),
        node(
            func=apply_gmm,
            inputs=["UMAP_Robust_data", "params:cluster_range"],
            outputs="robust_umap_gmm_results",
            name="robust_umap_gmm"
        ),
    ])
 
 
    return pipeline(nodes)