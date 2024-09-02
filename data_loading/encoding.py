import torch

# Create your dataset
class EncodingDataset(torch.utils.data.Dataset):
    def __init__(
            self, 
            tokenizer, 
            texts, 
            labels=None, 
            max_length=512,
            remove_duplicates=False
        ):

        if remove_duplicates:
            # print(f'Dataset with {len(texts)} samples before removing duplicates')
            texts_to_id = {text: i for i, text in enumerate(texts)}
            texts = list(texts_to_id.keys())
            labels = [labels[i] for i in texts_to_id.values()] if labels else None
            
        
        # print(f'Created dataset with {len(texts)} samples')

        self.inputs = tokenizer(
            texts, 
            return_tensors='pt', 
            truncation=True, 
            padding='max_length', 
            max_length=max_length
        )
        if labels:
            self.inputs['labels'] = torch.tensor(labels, dtype=torch.long) if labels else None

            print("Number of labels: ", len(set(labels)))
 

    def __len__(self):
        return len(self.inputs['input_ids'])
    

    def __getitem__(self, index):
        return {k: v[index] for k, v in self.inputs.items()}