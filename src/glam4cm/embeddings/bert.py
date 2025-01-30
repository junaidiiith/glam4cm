from transformers import AutoModel, AutoTokenizer
import torch
from typing import List, Union
from glam4cm.embeddings.common import Embedder
from glam4cm.data_loading.encoding import EncodingDataset
from torch.utils.data import DataLoader
from glam4cm.settings import device
import numpy as np

class BertEmbedder(Embedder):
    def __init__(self, model_name, ckpt=None):
        super().__init__(name='BERT')
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(ckpt if ckpt else model_name)
        self.model.to(device)
        self.finetuned = bool(ckpt)
    

    @property
    def embedding_dim(self) -> int:
        """
        Returns the embedding dimension of the model
        By Running the model on a dummy input and extracting the output shape
        """
        with torch.no_grad():
            dummy_input = self.tokenizer('Hello World!', return_tensors='pt')
            outputs = self.model(**{k: v.to(device) for k, v in dummy_input.items()})
            embedding_dim = outputs.last_hidden_state.shape[-1]
        return embedding_dim
    
    def embed(self, text: Union[str, List[str]], aggregate='cls'):
        torch.cuda.empty_cache()
        if isinstance(text, str):
            text = [text]
            
        print("Number of Texts: ", len(text))

        dataset = EncodingDataset(self.tokenizer, texts=text, remove_duplicates=False)
        loader = DataLoader(dataset, batch_size=256)

        embeddings = list()
        with torch.no_grad():
            for batch in loader:
                # Move inputs to device and process
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                
                outputs = self.model(input_ids, attention_mask)
                
                # Collect embeddings
                embeddings.append(outputs.last_hidden_state.cpu().numpy())
                
                # Free memory of the batch inputs
                del input_ids, attention_mask, outputs
                torch.cuda.empty_cache()  # Clear GPU memory
                

            # Combine all the embeddings
            embeddings = np.concatenate(embeddings, axis=0)
            
            if aggregate == 'mean':
                embeddings = np.mean(embeddings, axis=-1)
            elif aggregate == 'max':
                embeddings = np.max(embeddings, axis=-1)
            elif aggregate == 'cls':
                embeddings = embeddings[:, 0, :]
            else:
                raise ValueError(f'Unknown aggregation method: {aggregate}')
            
        # print("Embeddings Shape: ", embeddings.shape)
        assert embeddings.shape[0] == len(text)
        return embeddings