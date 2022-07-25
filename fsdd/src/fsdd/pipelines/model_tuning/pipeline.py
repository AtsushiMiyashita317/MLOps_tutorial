"""
This is a boilerplate pipeline 'model_tuning'
generated using Kedro 0.18.2
"""

from kedro.pipeline import Pipeline, node, pipeline
from .nodes import tuning


def create_pipeline(**kwargs) -> Pipeline:
    return pipeline([
        node(
            func=tuning,
            inputs=['dev_data', 'val_data', 'parameters'],
            outputs='best_params',
            name='tune_hyperparameters'
        )
    ])
