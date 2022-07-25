"""
This is a boilerplate pipeline 'feature_engineering'
generated using Kedro 0.18.2
"""

from kedro.pipeline import Pipeline, node, pipeline
from .nodes import split_data, feature_extruction


def create_pipeline(**kwargs) -> Pipeline:
    return pipeline([
        node(
            func=split_data,
            inputs=['train_annotations', 'params:val_rate'],
            outputs=['dev_annotations', 'val_annotations'],
            name='dev_val_split'
        ),
        node(
            func=feature_extruction,
            inputs='train_annotations',
            outputs='train_data',
            name='extruct_train'
        ),
        node(
            func=feature_extruction,
            inputs='test_annotations',
            outputs='test_data',
            name='extruct_test'
        ),
        node(
            func=feature_extruction,
            inputs='dev_annotations',
            outputs='dev_data',
            name='extruct_dev'
        ),
        node(
            func=feature_extruction,
            inputs='val_annotations',
            outputs='val_data',
            name='extruct_val'
        ),
    ])
