from collections import Counter
import numpy as np
from sklearn.model_selection import StratifiedKFold, train_test_split
from transformers import (
    Trainer, 
    TrainingArguments
)
from transformers import (
    AutoModelForSequenceClassification, 
    AutoTokenizer
)
from glam4cm.data_loading.encoding import EncodingDataset
from glam4cm.settings import device
from sklearn.preprocessing import LabelEncoder
from glam4cm.trainers.metrics import compute_metrics


class BertTrainer:
    def __init__(
        self,
        model_name,
        ckpt=None,
        max_length=512
    ):
        self.model_name = model_name
        self.ckpt = ckpt
        self.max_length = max_length


    def train(
        self,
        texts,
        labels,
        test_ratio=0.2,
        kfold=False,
        num_train_epochs=15,
        train_batch_size=2,
        eval_batch_size=128,
        weight_decay=0.01,
        logging_steps=50,
        eval_steps=50,
        save_steps=50,
        learning_rate=5e-5,
        warmup_steps=500,
        output_dir='./results',
        logs_dir='./logs',
        seed=42
    ):
        def train_fold():
            print(f'Train: {len(X_train)}, Test: {len(X_test)}')
            print("Class distribution in train: ", Counter(y_train))
            print("Class distribution in test: ", Counter(y_test))

            tokenizer = AutoTokenizer.from_pretrained(self.model_name if not self.ckpt else self.ckpt)
            model = AutoModelForSequenceClassification.from_pretrained(self.model_name, num_labels=num_classes)
            model.to(device)

            train_ds = EncodingDataset(tokenizer, X_train, y_train, max_length=self.max_length)
            test_ds = EncodingDataset(tokenizer, X_test, y_test, max_length=self.max_length)

            training_args = TrainingArguments(
                output_dir=output_dir,
                num_train_epochs=num_train_epochs,
                eval_strategy="steps",
                per_device_train_batch_size=train_batch_size,
                per_device_eval_batch_size=eval_batch_size,
                warmup_steps=warmup_steps,
                weight_decay=weight_decay,
                learning_rate=learning_rate,
                logging_dir=logs_dir,
                logging_steps=logging_steps,
                eval_steps=eval_steps,
                save_steps=save_steps,
                save_total_limit=2,
                load_best_model_at_end=True,
                fp16=True
            )

            trainer = Trainer(
                model=model,
                args=training_args,
                train_dataset=train_ds,
                eval_dataset=test_ds,
                compute_metrics=compute_metrics            
            )

            trainer.train()
            results = trainer.evaluate()
            print(results)


        y = LabelEncoder().fit_transform(labels)
        num_classes = len(set(y))
        if kfold > 0:
            k = int(1 / self.test_ratio)
            kfold = StratifiedKFold(n_splits=k, shuffle=True, random_state=seed)
            n = len(self.graphs)
            for i, (train_idx, test_idx) in enumerate(kfold.split(np.zeros(n), np.zeros(n))):
                X_train, y_train = [texts[i] for i in train_idx], [y[i] for i in train_idx]
                X_test, y_test = [texts[i] for i in test_idx], [y[i] for i in test_idx]
                print("Fold number: ", i+1)
                train_fold()
        else:
            X_train, X_test, y_train, y_test = train_test_split(texts, y, test_size=test_ratio, random_state=seed)
            train_fold()