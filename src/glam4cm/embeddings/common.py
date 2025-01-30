from abc import abstractmethod
import json
import os
from typing import List, Union
import torch
from glam4cm.settings import (
    WORD2VEC_MODEL,
    TFIDF_MODEL
)


class Embedder:
    def __init__(self, name: str):
        self.name = name
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
        from glam4cm.embeddings.bert import BertEmbedder
        return BertEmbedder(model_name, ckpt)
    elif WORD2VEC_MODEL in model_name:
        from glam4cm.embeddings.w2v import Word2VecEmbedder
        return Word2VecEmbedder()
    elif TFIDF_MODEL in model_name:
        from glam4cm.embeddings.tfidf import TfidfEmbedder
        return TfidfEmbedder()
    else:
        raise ValueError(f'Unknown model name: {model_name}')