"""
This is a boilerplate pipeline 'data_ingestion'
generated using Kedro 0.19.11
"""

from kedro.pipeline import node, Pipeline, pipeline  # noqa
from .nodes import just_reading


def create_pipeline(**kwargs) -> Pipeline:
    return pipeline([
        
        node(
            func=just_reading,
            inputs="Ecommerce",
            outputs="df_raw",
            name="Data_Ingestion"
        )
    ])
