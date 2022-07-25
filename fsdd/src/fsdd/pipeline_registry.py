"""Project pipelines."""
from typing import Dict

from kedro.pipeline import Pipeline, pipeline

import fsdd.pipelines.feature_engineering as fe
import fsdd.pipelines.model_tuning as tuning
import fsdd.pipelines.model_testing as testing

def register_pipelines() -> Dict[str, Pipeline]:
    """Register the project's pipelines.

    Returns:
        A mapping from a pipeline name to a ``Pipeline`` object.
    """
    fe_pipeline = fe.create_pipeline()
    tuning_pipeline = tuning.create_pipeline()
    testing_pipeline = testing.create_pipeline()
    
    return {
        "__default__": fe_pipeline + tuning_pipeline + testing_pipeline,
        "fe": fe_pipeline,
        "tuning": tuning_pipeline,
        "test": testing_pipeline
    }
