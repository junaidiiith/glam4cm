from transformers import (
    Trainer, 
    TrainingArguments
)
from transformers import (
    AutoModelForSequenceClassification, 
    AutoTokenizer
)
from data_loading.graph_dataset import GraphDataset
from data_loading.models_dataset import EcoreModelDataset
from settings import device
from sklearn.preprocessing import LabelEncoder
from trainers.metrics import compute_metrics


max_length_map = {
    'bert-base-uncased': 512,
    'allenai/longformer-base-4096': 4096
}

class BertTrainer:
    def __init__(
        self,
        dataset: GraphDataset,
        model_name,
        num_epochs=3
    ):
        self.dataset = dataset
        self.model_name = model_name
        self.num_epochs = num_epochs

        dataset = dataset.get_lm_graph_classification_data() 


def train_hf(model_name, model_ds: EcoreModelDataset, epochs):
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

        dataset_name = model_ds.name
        output_dir = f'./results/{dataset_name}'
        logs_dir = f'./logs/{dataset_name}'

        # Training arguments
        training_args = TrainingArguments(
            output_dir=output_dir,
            num_train_epochs=epochs,
            eval_strategy="epoch",
            per_device_train_batch_size=2,
            per_device_eval_batch_size=2,
            weight_decay=0.01,
            logging_dir=logs_dir,
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