"""
This is a boilerplate pipeline 'data_processing'
generated using Kedro 0.18.4
"""

from kedro.pipeline import Pipeline, node, pipeline
from .nodes import preprocess_ecommerce, feature_filtering
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
            node(
                func=feature_filtering,
                inputs="preprocessed_Ecommerce",               
                outputs=dict(
                    filtered_Ecommerce="filtered_Ecommerce", 
                    returns="df_returns", 
                    purchases="df_purchases",
                    ),
                name="Data_Filtering"                       
            ),
        ])
