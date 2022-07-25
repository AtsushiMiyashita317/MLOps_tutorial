"""
This is a boilerplate pipeline 'feature_engineering'
generated using Kedro 0.18.2
"""

import pandas as pd
import torch

def split_data(annotations:pd.DataFrame, val_rate:float):
    return annotations, annotations

def feature_extruction(annotations:pd.DataFrame):
    return torch.Tensor()
