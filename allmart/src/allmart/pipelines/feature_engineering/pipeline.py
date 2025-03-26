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
                inputs=["df_preprocessed", "df_rfm"],               
                outputs="df_engineered",
                name="Feature_Engineering"                       
            ),
            # node(
            #     func=feature_normalization,
            #     inputs="df_engineered",               
            #     outputs="normalized_Ecommerce",
            #     name="Feature_Normalization"                       
            # ),
])