import torch

# Create your dataset
class EncodingDataset(torch.utils.data.Dataset):
    def __init__(self, tokenizer, texts, labels=None, max_length=512):
        self.inputs = tokenizer(
            texts, 
            return_tensors='pt', 
            truncation=True, 
            padding='max_length', 
            max_length=max_length
        )
        if labels:
            self.inputs['labels'] = torch.tensor(labels, dtype=torch.long)
 

    def __len__(self):
        return len(self.inputs['input_ids'])
    

    def __getitem__(self, index):
        item = {key: val[index] for key, val in self.inputs.items()}
        return item
