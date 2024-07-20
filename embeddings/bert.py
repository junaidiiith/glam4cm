from transformers import AutoModel, AutoTokenizer
import torch
from typing import List, Union
from embeddings.common import Embedder
from data_loading.dataset import EncodingDataset
from torch.utils.data import DataLoader
from settings import device


class BertEmbedder(Embedder):
    def __init__(self, model_name, ckpt=None):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(ckpt if ckpt else model_name)
    
    def embed(self, text: Union[str, List[str]], aggregate='mean'):
        dataset = EncodingDataset(self.tokenizer, texts=text)
        loader = DataLoader(dataset, batch_size=128)

        with torch.no_grad():
            embeddings = []
            for batch in loader:
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                outputs = self.model(input_ids, attention_mask)
                embeddings.append(outputs.last_hidden_state)
                
            
            embeddings = torch.cat(embeddings, dim=0)
            if aggregate == 'mean':
                embeddings = embeddings.mean(dim=1)
            elif aggregate == 'max':
                embeddings = embeddings.max(dim=1)
            elif aggregate == 'cls':
                embeddings = embeddings[:, 0, :]
            elif aggregate == 'pool':
                embeddings = embeddings.mean(dim=1)
            else:
                raise ValueError(f'Unknown aggregation method: {aggregate}') 
