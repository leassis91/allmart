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
)


def create_pipeline(**kwargs) -> Pipeline:
    
    # --- 1. DATA PREPROCESSING ---
    nodes = []
    # Step 1: Data Scaling
    nodes.append(
        node(
            func=scale_data,
            inputs=["Engineered_features", "RFM_features", "params:features"],
            outputs=["Standardized_data", "Normalized_data", "Robust_data"],
            name="data_scaling"
        )
    )
    # # Step 2: Dimensionality Reduction
    # nodes.append(
    #     node(
    #         func=reduce_data_with_pca,
    #         inputs=["scaled_data"],
    #         outputs="pca_data",
    #         name="reduce_data_with_pca"
    #     )
    # )
    # nodes.append(
    #     node(
    #         func=reduce_data_with_umap,
    #         inputs=["scaled_data"],
    #         outputs="umap_data",
    #         name="reduce_data_with_umap"
    #     )
    # )
    # # Step 2: Dimensionality Reduction
    # nodes.append(
    #     node(
    #         func=reduce_data_with_pca,
    #         inputs=["scaled_data"],
    #         outputs="standard_pca_data",
    #         name="reduce_data_with_standard_pca"
    #     )
    # )
    # nodes.append(
    #     node(
    #         func=reduce_data_with_umap,
    #         inputs=["scaled_data"],
    #         outputs="standard_umap_data",
    #         name="reduce_data_with_standard_umap"
    #     )
    # )
    # nodes.append(
    #     node(
    #         func=reduce_data_with_pca,
    #         inputs=["scaled_data"],
    #         outputs="minmax_pca_data",
    #         name="reduce_data_with_minmax_pca"
    #     )
    # )
    # nodes.append(
    #     node(
    #         func=reduce_data_with_umap,
    #         inputs=["scaled_data"],
    #         outputs="minmax_umap_data",
    #         name="reduce_data_with_minmax_umap"
    #     )
    # )
    # nodes.append(
    #     node(
    #         func=reduce_data_with_pca,
    #         inputs=["scaled_data"],
    #         outputs="robust_pca_data",
        
    # # Step 3: Clustering and Evaluation
    # model_clusters = []  # To collect all model result nodes for final combination
    
    # # Standard Scaler + PCA combinations
    # standard_pca_kmeans_output = "standard_pca_kmeans_results"
    # model_clusters.append(standard_pca_kmeans_output)
    # nodes.append(
    #     node(
    #         func=apply_kmeans,
    #         inputs=["X_reduced_pca", "params:cluster_range"],
    #         outputs=standard_pca_kmeans_output,
    #         name="standard_pca_kmeans"
    #     )
    # )
    
    # standard_pca_hierarchical_output = "standard_pca_hierarchical_results"
    # model_clusters.append(standard_pca_hierarchical_output)
    # nodes.append(
    #     node(
    #         func=apply_standard_pca_hierarchical,
    #         inputs=["standard_pca_data", "params:cluster_range"],
    #         outputs=standard_pca_hierarchical_output,
    #         name="standard_pca_hierarchical"
    #     )
    # )
    
    # standard_pca_dbscan_output = "standard_pca_dbscan_results"
    # model_clusters.append(standard_pca_dbscan_output)
    # nodes.append(
    #     node(
    #         func=apply_standard_pca_dbscan,
    #         inputs=["standard_pca_data", "params:eps_range"],
    #         outputs=standard_pca_dbscan_output,
    #         name="standard_pca_dbscan"
    #     )
    # )
    
    # standard_pca_gmm_output = "standard_pca_gmm_results"
    # model_clusters.append(standard_pca_gmm_output)
    # nodes.append(
    #     node(
    #         func=apply_standard_pca_gmm,
    #         inputs=["standard_pca_data", "params:cluster_range"],
    #         outputs=standard_pca_gmm_output,
    #         name="standard_pca_gmm"
    #     )
    # )
    
    # # Standard Scaler + UMAP combinations
    # standard_umap_kmeans_output = "standard_umap_kmeans_results"
    # model_clusters.append(standard_umap_kmeans_output)
    # nodes.append(
    #     node(
    #         func=apply_standard_umap_kmeans,
    #         inputs=["standard_umap_data", "params:cluster_range"],
    #         outputs=standard_umap_kmeans_output,
    #         name="standard_umap_kmeans"
    #     )
    # )
    
    # standard_umap_hierarchical_output = "standard_umap_hierarchical_results"
    # model_clusters.append(standard_umap_hierarchical_output)
    # nodes.append(
    #     node(
    #         func=apply_standard_umap_hierarchical,
    #         inputs=["standard_umap_data", "params:cluster_range"],
    #         outputs=standard_umap_hierarchical_output,
    #         name="standard_umap_hierarchical"
    #     )
    # )
    
    # standard_umap_dbscan_output = "standard_umap_dbscan_results"
    # model_clusters.append(standard_umap_dbscan_output)
    # nodes.append(
    #     node(
    #         func=apply_standard_umap_dbscan,
    #         inputs=["standard_umap_data", "params:eps_range"],
    #         outputs=standard_umap_dbscan_output,
    #         name="standard_umap_dbscan"
    #     )
    # )
    
    # standard_umap_gmm_output = "standard_umap_gmm_results"
    # model_clusters.append(standard_umap_gmm_output)
    # nodes.append(
    #     node(
    #         func=apply_standard_umap_gmm,
    #         inputs=["standard_umap_data", "params:cluster_range"],
    #         outputs=standard_umap_gmm_output,
    #         name="standard_umap_gmm"
    #     )
    # )
    
    # # MinMax Scaler + PCA combinations
    # minmax_pca_kmeans_output = "minmax_pca_kmeans_results"
    # model_clusters.append(minmax_pca_kmeans_output)
    # nodes.append(
    #     node(
    #         func=apply_minmax_pca_kmeans,
    #         inputs=["minmax_pca_data", "params:cluster_range"],
    #         outputs=minmax_pca_kmeans_output,
    #         name="minmax_pca_kmeans"
    #     )
    # )
    
    # minmax_pca_hierarchical_output = "minmax_pca_hierarchical_results"
    # model_clusters.append(minmax_pca_hierarchical_output)
    # nodes.append(
    #     node(
    #         func=apply_minmax_pca_hierarchical,
    #         inputs=["minmax_pca_data", "params:cluster_range"],
    #         outputs=minmax_pca_hierarchical_output,
    #         name="minmax_pca_hierarchical"
    #     )
    # )
    
    # minmax_pca_dbscan_output = "minmax_pca_dbscan_results"
    # model_clusters.append(minmax_pca_dbscan_output)
    # nodes.append(
    #     node(
    #         func=apply_minmax_pca_dbscan,
    #         inputs=["minmax_pca_data", "params:eps_range"],
    #         outputs=minmax_pca_dbscan_output,
    #         name="minmax_pca_dbscan"
    #     )
    # )
    
    # minmax_pca_gmm_output = "minmax_pca_gmm_results"
    # model_clusters.append(minmax_pca_gmm_output)
    # nodes.append(
    #     node(
    #         func=apply_minmax_pca_gmm,
    #         inputs=["minmax_pca_data", "params:cluster_range"],
    #         outputs=minmax_pca_gmm_output,
    #         name="minmax_pca_gmm"
    #     )
    # )
    
    # # MinMax Scaler + UMAP combinations
    # minmax_umap_kmeans_output = "minmax_umap_kmeans_results"
    # model_clusters.append(minmax_umap_kmeans_output)
    # nodes.append(
    #     node(
    #         func=apply_minmax_umap_kmeans,
    #         inputs=["minmax_umap_data", "params:cluster_range"],
    #         outputs=minmax_umap_kmeans_output,
    #         name="minmax_umap_kmeans"
    #     )
    # )
    
    # minmax_umap_hierarchical_output = "minmax_umap_hierarchical_results"
    # model_clusters.append(minmax_umap_hierarchical_output)
    # nodes.append(
    #     node(
    #         func=apply_minmax_umap_hierarchical,
    #         inputs=["minmax_umap_data", "params:cluster_range"],
    #         outputs=minmax_umap_hierarchical_output,
    #         name="minmax_umap_hierarchical"
    #     )
    # )
    
    # minmax_umap_dbscan_output = "minmax_umap_dbscan_results"
    # model_clusters.append(minmax_umap_dbscan_output)
    # nodes.append(
    #     node(
    #         func=apply_minmax_umap_dbscan,
    #         inputs=["minmax_umap_data", "params:eps_range"],
    #         outputs=minmax_umap_dbscan_output,
    #         name="minmax_umap_dbscan"
    #     )
    # )
    
    # minmax_umap_gmm_output = "minmax_umap_gmm_results"
    # model_clusters.append(minmax_umap_gmm_output)
    # nodes.append(
    #     node(
    #         func=apply_minmax_umap_gmm,
    #         inputs=["minmax_umap_data", "params:cluster_range"],
    #         outputs=minmax_umap_gmm_output,
    #         name="minmax_umap_gmm"
    #     )
    # )
    
    # # Robust Scaler + PCA combinations
    # robust_pca_kmeans_output = "robust_pca_kmeans_results"
    # model_clusters.append(robust_pca_kmeans_output)
    # nodes.append(
    #     node(
    #         func=apply_robust_pca_kmeans,
    #         inputs=["robust_pca_data", "params:cluster_range"],
    #         outputs=robust_pca_kmeans_output,
    #         name="robust_pca_kmeans"
    #     )
    # )
    
    # robust_pca_hierarchical_output = "robust_pca_hierarchical_results"
    # model_clusters.append(robust_pca_hierarchical_output)
    # nodes.append(
    #     node(
    #         func=apply_robust_pca_hierarchical,
    #         inputs=["robust_pca_data", "params:cluster_range"],
    #         outputs=robust_
    
    # # --- 4. RESULTS COMBINATION AND RANKING ---
    # # Final node to combine and rank all clustering results
    # nodes.append(
    #     node(
    #         func=combine_and_rank_results,
    #         inputs=model_clusters,
    #         outputs="best_results",
    #         name="combine_and_rank_results"
    #     )
    # )

    
    return pipeline(nodes)