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
    # Create pipeline instances
    data_ingestion_pipeline = data_ingestion.create_pipeline()
    data_preprocessing_pipeline = data_preprocessing.create_pipeline()
    feature_engineering_pipeline = feature_engineering.create_pipeline()
    model_training_pipeline = model_training.create_pipeline()
    model_prediction_pipeline = model_prediction.create_pipeline()
    
    # Register pipelines
    return {
        "__default__": (
            data_ingestion_pipeline + 
            data_preprocessing_pipeline + 
            feature_engineering_pipeline + 
            model_training_pipeline + 
            model_prediction_pipeline
        ),
        "data_ingestion": data_ingestion_pipeline,
        "data_preprocessing": data_preprocessing_pipeline,
        "feature_engineering": feature_engineering_pipeline,
        "data_eng": data_preprocessing_pipeline + feature_engineering_pipeline,
        "model_training": model_training_pipeline,
        "model_prediction": model_prediction_pipeline,
        "data_science": (
            data_preprocessing_pipeline + 
            feature_engineering_pipeline + 
            model_training_pipeline + 
            model_prediction_pipeline
        ),
    }