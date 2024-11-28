import torch
from typing import List, Union
from embeddings.common import Embedder
from settings import W2V_CONFIG
from gensim.models import Word2Vec


class Word2VecEmbedder(Embedder):
    def __init__(self):
        super().__init__(name='Word2Vec')

    @property
    def embedding_dim(self) -> int:
        return self.model.vector_size
    
    def train(self, texts: List[str]):
        print("Word2VecEmbedder: Training Word2Vec model")
        self.model = Word2Vec(texts, **W2V_CONFIG)
        print("Word2VecEmbedder: Word2Vec model trained")
    
    def embed(self, text: Union[str, List[str]]):
        if isinstance(text, str):
            text = text.split()
        word_vectors = [self.model.wv[word] for word in text if word in self.model.wv]
        if word_vectors:
            return torch.tensor(word_vectors).mean(dim=0)
        else:
            return torch.zeros(self.embedding_dim)