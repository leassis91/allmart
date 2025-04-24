"""
This is a boilerplate pipeline 'data_preprocessing'
generated using Kedro 0.19.11
"""

from kedro.pipeline import node, Pipeline, pipeline  # noqa

from .nodes import preprocess_ecommerce



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
                inputs="df_raw",               
                outputs="df_preprocessed",
                name="Data_Preprocessing"                       
            ),
        ])

