from sklearn.feature_extraction.text import TfidfVectorizer
from transformers import AutoTokenizer
from sklearn.preprocessing import LabelEncoder
import fasttext
from scipy.sparse import csr_matrix
import numpy as np
from encoding.common import (
    doc_tokenizer, 
    SEP
)


class TFIDFEncoder:
    def __init__(self, X=None):
        self.encoder = TfidfVectorizer(
            lowercase=False, tokenizer=doc_tokenizer, min_df=3
        )

        if X:
            self.encode(X)

    def encode(self, X):
        # print('Fitting TFIDF')
        X_t = self.encoder.fit_transform(X)
        X_sp = csr_matrix(np.vstack([x.toarray() for x in X_t]))
        # print('TFIDF Encoded')
        return X_sp


class BertTokenizerEncoder:
    def __init__(self, name, X=None):
        self.tokenizer = AutoTokenizer.from_pretrained(name)

        if X:
            self.encode(X)
        

    def encode(self, X, batch_encode=False, percentile=100):
        # print('Tokenizing Bert')
        tokens = self.tokenizer(X)

        if batch_encode:
            lengths = [len(i) for i in tokens['input_ids']]
            size = int(np.percentile(lengths, percentile)) if percentile < 100 else max(lengths)
            if size > 512:
                print(f'WARNING: Max size is {size}. Truncating to 512')
            size = max(size, 512)
            
            tokenized_data = self.tokenizer(
                X, 
                padding=True, 
                truncation=True, 
                max_length=size
            )
        else:
            tokenized_data = self.tokenizer(X)
        # print('Bert Tokenized')

        return tokenized_data


class BertTFIDF:
    def __init__(self, name, X=None):
        self.bert = BertTokenizerEncoder(name)
        self.tfidf = TFIDFEncoder()

        if X:
            self.encode(X)

    def encode(self, X):
        X_b = [f"{SEP}".join([str(j) for j in i]) for i in self.bert.encode(X)['input_ids']]
        X_t = self.tfidf.encode(X_b)
        return X_t


class FasttextEncoder:
    def __init__(self, model_name, X=None):
        self.model = fasttext.load_model(model_name)
        if X:
            self.encode(X)

    def encode(self, X):
        def get_sentence_embedding(sentence):
            return self.model.get_sentence_vector(sentence)
        
        # print('Encoding Fasttext')
        X_t = [" ".join(doc_tokenizer(i)) for i in X]
        X_t = np.array([get_sentence_embedding(i) for i in X_t])
        # print('Fasttext Encoded')
        return X_t


class ClassLabelEncoder(LabelEncoder):
    def __init__(self, y=None) -> None:
        super().__init__()
        if y:
            self.fit(y)
    
    def encode(self, y):
        return self.fit_transform(y)