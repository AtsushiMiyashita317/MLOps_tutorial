import os
from kedro.io import AbstractDataSet
import torch

class TensorDataSet(AbstractDataSet):
    def __init__(self, filepath):
        super().__init__()
        self._filepath = filepath

    def _load(self):
        return torch.load(self._filepath)
    
    def _save(self, data) -> None:
        torch.save(data, self._filepath)
     
    def _exists(self) -> bool:
        return os.path.exists(self._filepath)
    
    def _describe(self):
        return dict(filepath=self._filepath)