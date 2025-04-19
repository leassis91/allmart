"""
This is a boilerplate pipeline 'feature_engineering'
generated using Kedro 0.19.11
"""

from kedro.pipeline import node, Pipeline, pipeline  # noqa

from .nodes import feature_engineering

def create_pipeline(**kwargs) -> Pipeline:
    return pipeline([

            node(
                func=feature_engineering,
                inputs="df_preprocessed",               
                outputs="RFM_features",
                name="Feature_Engineering"                       
            ),
])