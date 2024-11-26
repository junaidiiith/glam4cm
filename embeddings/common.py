from abc import abstractmethod
import json
import os
from typing import List, Union
import torch


class Embedder:

    def ___init__(self):
        self.finetuned = False

    @abstractmethod
    def embed(self, text: Union[str, List[str]], aggregate='mean') -> torch.Tensor:
        pass

    @property
    def embedding_dim(self) -> int:
        pass


def get_embedding_model(
        model_name: str,
        ckpt: str = None
    ) -> Embedder:
    if ckpt:
        model_name = json.load(open(os.path.join(ckpt, 'config.json')))['_name_or_path']
        
    if 'bert' in model_name:
        from embeddings.bert import BertEmbedder
        return BertEmbedder(model_name, ckpt)
    else:
        raise ValueError(f'Unknown model name: {model_name}')