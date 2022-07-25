"""
This is a boilerplate pipeline 'model_testing'
generated using Kedro 0.18.2
"""

from kedro.pipeline import Pipeline, node, pipeline
from .nodes import training

def create_pipeline(**kwargs) -> Pipeline:
    return pipeline([
        node(
            func=training,
            inputs=['train_data', 'test_data', 'parameters', 'best_params'],
            outputs='model',
            name='training'
        )
    ])
