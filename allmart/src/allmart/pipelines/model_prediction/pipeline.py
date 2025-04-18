"""
This is a boilerplate pipeline 'model_prediction'
generated using Kedro 0.19.11
"""

from kedro.pipeline import node, Pipeline, pipeline  # noqa

from .nodes import (
    combine_results,
    find_best_model,
    generate_cluster_report,
    )

def create_pipeline(**kwargs) -> Pipeline:
           
    nodes = []
    
    # --- 4. FIND BEST MODEL NODE ---
    
    # Collector nodes for each algorithm type
    nodes.append(
        node(
            func=lambda *args: list(args),
            inputs=[
                "standard_pca_kmeans_results", "standard_umap_kmeans_results",
                "minmax_pca_kmeans_results", "minmax_umap_kmeans_results",
                "robust_pca_kmeans_results", "robust_umap_kmeans_results"
            ],
            outputs="kmeans_results",
            name="collect_kmeans_results"
        )
    )

    nodes.append(
        node(
            func=lambda *args: list(args),
            inputs=[
                "standard_pca_hierarchical_results", "standard_umap_hierarchical_results",
                "minmax_pca_hierarchical_results", "minmax_umap_hierarchical_results",
                "robust_pca_hierarchical_results", "robust_umap_hierarchical_results"
            ],
            outputs="hierarchical_results",
            name="collect_hierarchical_results"
        )
    )

    nodes.append(
        node(
            func=lambda *args: list(args),
            inputs=[
                "standard_pca_dbscan_results", "standard_umap_dbscan_results",
                "minmax_pca_dbscan_results", "minmax_umap_dbscan_results",
                "robust_pca_dbscan_results", "robust_umap_dbscan_results"
            ],
            outputs="dbscan_results",
            name="collect_dbscan_results"
        )
    )

    nodes.append(
        node(
            func=lambda *args: list(args),
            inputs=[
                "standard_pca_gmm_results", "standard_umap_gmm_results",
                "minmax_pca_gmm_results", "minmax_umap_gmm_results",
                "robust_pca_gmm_results", "robust_umap_gmm_results"
            ],
            outputs="gmm_results",
            name="collect_gmm_results"
        )
    )
    
    
    nodes.append(
    node(
        func=combine_results,
        inputs=["kmeans_results", "hierarchical_results", "dbscan_results", "gmm_results"],
        outputs="combined_model_results",
        name="combine_model_results"
    )
)

    nodes.append(
        node(
            func=find_best_model,
            inputs="combined_model_results",
            outputs="best_model",
            name="select_best_model"
        )
    )

    nodes.append(
        node(
            func=generate_cluster_report,
            inputs=["df_original", "best_model"],
            outputs="cluster_report",
            name="generate_cluster_report"
        )
    )

    return pipeline(nodes)