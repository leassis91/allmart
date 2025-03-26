"""Project pipelines."""

from kedro.framework.project import find_pipelines
from kedro.pipeline import Pipeline
from typing import Dict

from allmart.pipelines import (data_ingestion, 
                              data_preprocessing, 
                              feature_engineering, 
                              model_training, 
                              model_prediction
)



def register_pipelines() -> Dict[str, Pipeline]:
    """Register the project's pipelines.

    Returns:
        A mapping from pipeline names to ``Pipeline`` objects.
    """
    
    DI = data_ingestion.create_pipeline()
    DP = data_preprocessing.create_pipeline()
    FE = feature_engineering.create_pipeline()
    MT = model_training.create_pipeline()
    MP = model_prediction.create_pipeline()
    
    pipelines = find_pipelines()
    

    return {
        "__default__": sum(pipelines.values()),
        "ingestao": DI,
        "processamento": DP,
        "feature_engineering": FE,
        "data_eng": DP+FE,
        "model_training": MT,
        "model_prediction": MP,
        "data_science": DP+FE+MT+MP,
    }