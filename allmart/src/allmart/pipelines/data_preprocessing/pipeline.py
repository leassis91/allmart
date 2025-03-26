"""
This is a boilerplate pipeline 'data_preprocessing'
generated using Kedro 0.19.11
"""

from kedro.pipeline import node, Pipeline, pipeline  # noqa

from .nodes import preprocess_ecommerce, create_rfm



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
                outputs=["df_preprocessed", "df_rfm"],
                name="Data_Preprocessing"                       
            ),
            # node(
            #     func=create_rfm,
            #     inputs="df_preprocessed",               
            #     outputs="df_rfm",
            #     name="RFM_Features"                       
            # ),
            # node(
            #     func=feature_normalization,
            #     inputs="feature_engineered_Ecommerce",               
            #     outputs="normalized_Ecommerce",
            #     name="Feature_Normalization"                       
            # ),
        ])

