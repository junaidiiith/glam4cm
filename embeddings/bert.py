from transformers import AutoModel, AutoTokenizer
import torch
from typing import List, Union
from embeddings.common import Embedder
from data_loading.encoding import EncodingDataset
from torch.utils.data import DataLoader
from settings import device


class BertEmbedder(Embedder):
    def __init__(self, model_name, ckpt=None):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(ckpt if ckpt else model_name)
        self.model.to(device)
        self.finetuned = bool(ckpt)
    
    def embed(self, text: Union[str, List[str]], aggregate='cls'):
        dataset = EncodingDataset(self.tokenizer, texts=text, remove_duplicates=False)
        loader = DataLoader(dataset, batch_size=128)

        with torch.no_grad():
            embeddings = []
            for batch in loader:
                outputs = self.model(
                    batch['input_ids'].to(device), 
                    batch['attention_mask'].to(device)
                )
                embeddings.append(outputs.last_hidden_state.cpu())
                
            
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
        
        return embeddings
