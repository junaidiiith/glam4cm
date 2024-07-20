from abc import abstractmethod
from typing import List, Union


class Embedder:

    def ___init__(self):
        self.finetuned = False

    @abstractmethod
    def embed(self, text: Union[str, List[str]], aggregate='mean'):
        pass