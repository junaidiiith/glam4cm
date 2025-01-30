from typing import List, Union
from embeddings.common import Embedder
from glam4cm.settings import W2V_CONFIG
from gensim.models import Word2Vec
import numpy as np


class Word2VecEmbedder(Embedder):
    def __init__(self):
        super().__init__(name='Word2Vec')

    @property
    def embedding_dim(self) -> int:
        return self.model.vector_size
    
    def train(self, texts: List[str]):
        print("Word2VecEmbedder: Training Word2Vec model")
        texts = [text.split() for text in texts]
        self.model = Word2Vec(texts, **W2V_CONFIG, epochs=100)
        print("Total words in the model:", len(self.model.wv))
        print("Word2VecEmbedder: Word2Vec model trained")
    
    def embed(self, text: Union[str, List[str]]):

        def get_text_embedding(text: str):
            words = text.split()
            word_vectors = [self.model.wv[word] for word in words if word in self.model.wv]
            if word_vectors:
                return np.mean(word_vectors, axis=0)
            else:
                return np.zeros(self.embedding_dim)

        if isinstance(text, str):
            text = [text]
        word_vectors = [get_text_embedding(t) for t in text]
        return np.array(word_vectors)
        