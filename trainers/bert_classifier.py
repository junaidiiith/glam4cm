from transformers import (
    Trainer, 
    TrainingArguments
)
from transformers import (
    AutoModelForSequenceClassification, 
    AutoTokenizer
)
import torch
import numpy as np
import random
from data_loading.data import ModelDataset
from settings import device, seed
from sklearn.preprocessing import LabelEncoder
from trainers.metrics import compute_metrics


random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)

max_length_map = {
    'bert-base-uncased': 512,
    'allenai/longformer-base-4096': 4096
}

# Create your dataset
class CustomDataset(torch.utils.data.Dataset):
    def __init__(self, texts, labels, tokenizer, max_length):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        label = self.labels[idx]
        inputs = self.tokenizer(
            text, 
            return_tensors='pt', 
            truncation=True, 
            padding='max_length', 
            max_length=self.max_length
        )
        inputs = {k: v.squeeze(0) for k, v in inputs.items()}
        inputs['labels'] = torch.tensor(label, dtype=torch.long)
        return inputs


def train_hf(model_name, model_ds: ModelDataset, epochs):
    max_len = max_length_map[model_name]
    i = 0
    print(f'Device used: {device}')

    for train_idx, test_idx in model_ds.k_fold_split():
        print(f'Fold number: {i+1}')
        X, y = model_ds.data
        print(f'X: {len(X)}, y: {len(y)}')
        y = LabelEncoder().fit_transform(y)
        X_train, X_test = [X[i] for i in train_idx], [X[i] for i in test_idx]
        y_train, y_test = [y[i] for i in train_idx], [y[i] for i in test_idx]

        print(f'Train: {len(X_train)}, Test: {len(X_test)}')


        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=len(set(y)))
        model.to(device)

        train_ds = CustomDataset(X_train, y_train, tokenizer, max_length=max_len)
        test_ds = CustomDataset(X_test, y_test, tokenizer, max_length=max_len)

        # Training arguments
        training_args = TrainingArguments(
            output_dir='./results',
            num_train_epochs=epochs,
            eval_strategy="epoch",
            per_device_train_batch_size=2,
            per_device_eval_batch_size=2,
            warmup_steps=500,
            weight_decay=0.01,
            logging_dir='./logs',
            logging_steps=10,
        )

        # Trainer
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=train_ds,
            eval_dataset=test_ds,
            compute_metrics=compute_metrics            
        )

        # Train the model
        trainer.train()
        results = trainer.evaluate()
        print(results)

        i += 1