import numpy as np
from typing import List, Union
from glam4cm.embeddings.common import Embedder
from sklearn.feature_extraction.text import TfidfVectorizer


class TfidfEmbedder(Embedder):
    def __init__(self):
        super().__init__(name='TFIDF')
        pass

    def train(self, texts: List[str]):
        print("TFIDFEmbedder: Training TF-IDF model")
        self.model = TfidfVectorizer()
        self.model.fit(texts)
        print("TFIDFEmbedder: Model trained")

    @property
    def embedding_dim(self) -> int:
        return len(self.model.get_feature_names_out())
    
    def embed(self, text: Union[str, List[str]]):
        if isinstance(text, str):
            text = [text]
        return self.model.transform(text)