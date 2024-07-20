from abc import abstractmethod
from typing import List, Union
import torch


class Embedder:

    def ___init__(self):
        self.finetuned = False

    @abstractmethod
    def embed(self, text: Union[str, List[str]], aggregate='mean') -> torch.Tensor:
        pass
