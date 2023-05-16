"""
This is a boilerplate pipeline 'data_processing'
generated using Kedro 0.18.4
"""

from kedro.pipeline import Pipeline, node, pipeline
from .nodes import preprocess_ecommerce
from typing import Dict, Tuple


def create_pipeline(**kwargs) -> Pipeline:
    """Create the project's pipeline.
    
    Args:
        kwargs: Ignore any additional arguments added in the future.
        
    Returns:
        A mapping from a pipeline name to a ``Pipeline`` object.
    """
    return pipeline(
        [
            node(
                func=preprocess_ecommerce,
                inputs="Ecommerce",               
                outputs="preprocessed_Ecommerce",
                name="Data_Cleaning"                       
            ),
        ])
