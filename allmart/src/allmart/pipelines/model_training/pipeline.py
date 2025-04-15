"""
This is a boilerplate pipeline 'model_training'
generated using Kedro 0.19.11
"""

from kedro.pipeline import node, Pipeline, pipeline  # noqa

# from .nodes import split_train, making_preds
from .nodes import (
    scale_data,
    reduce_dimensionality,
    find_best_models,
    # split_train,
    # making_preds
)

# def create_pipeline(**kwargs) -> Pipeline:
#     return pipeline([
#         node(
#             split_train,
#             inputs="df_engineered",
#             outputs=["X_train"],
#             name="split_data"
#         ),
#         node(
#             making_preds,
#             inputs=["X_train", "params:features", "params:component_range", "params:cluster_range", "params:eps_range", "params:n_estimators"],
#             outputs="best_results",
#             name="model_training_pipeline"
#         ),
#     ])


def create_pipeline(**kwargs) -> Pipeline:
    return pipeline([
        # Step 1: Data Scaling
        node(
            func=lambda X, features: [
                {'id': f'scaled_{i}', 'scaler': scaler, 'data': scale_data(X, features, scaler)}
                for i, scaler in enumerate(['StandardScaler', 'MinMaxScaler', 'RobustScaler'])
            ],
            inputs=["df_engineered", "params:features"],
            outputs="scaled_data_list",
            name="data_scaling"
        ),
        
        # Step 2: Dimensionality Reduction
        node(
            func=lambda scaled_data_list, component_range, n_estimators: [
                {'scaler_id': data_dict['id'], 
                 'reductor': reductor, 
                 'n_components': n_comp,
                 'data': reduce_dimensionality(data_dict['data'], None, reductor, n_comp, n_estimators)}
                for data_dict in scaled_data_list
                for reductor in ['PCA', 'UMAP']  # Excluding TreeBasedEmbedding as it requires y
                for n_comp in range(component_range["start"], component_range["stop"], component_range.get("step", 1))
            ],
            inputs=["scaled_data_list", "params:component_range", "params:n_estimators"],
            outputs="reduced_data_list",
            name="dimensionality_reduction"
        ),
        
        # Step 3: Clustering and Evaluation
        node(
            func=find_best_models,
            inputs=["scaled_data_list", "reduced_data_list", 
                   "params:model_types", "params:cluster_range", "params:eps_range"],
            outputs="clustering_results",
            name="clustering_evaluation"
        ),
        
        # Step 4: Select Best Models
        node(
            func=lambda results: results.sort_values(by="silhouette", ascending=False).head(10),
            inputs="clustering_results",
            outputs="best_results",
            name="select_best_models"
        ),
    ])