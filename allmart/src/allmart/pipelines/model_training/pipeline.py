"""
This is a boilerplate pipeline 'model_training'
generated using Kedro 0.19.11
"""

from kedro.pipeline import node, Pipeline, pipeline  # noqa

from .nodes import split_train, making_preds

def create_pipeline(**kwargs) -> Pipeline:
    return pipeline([
        node(
            split_train,
            inputs="df_engineered",
            outputs="X_train",
            name="split_train_set"
        ),
        node(
            making_preds,
            inputs="X_train",
            outputs="model_trained",
            name="model_training"
        ),
    ])
