import os
import json
from argparse import ArgumentParser
from random import shuffle

import numpy as np
from sklearn.model_selection import StratifiedKFold

from sklearn.metrics import (
    accuracy_score, 
    balanced_accuracy_score, 
    f1_score, 
    recall_score
)
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    Trainer,
    TrainingArguments
)

from data_loading.encoding import EncodingDataset



def compute_metrics(pred):
    labels = pred.label_ids
    preds = np.argmax(pred.predictions, axis=1)
    acc = (preds == labels).mean()
    f1_macro = f1_score(labels, preds, average='macro')
    accuracy = accuracy_score(labels, preds)
    recall = recall_score(labels, preds, average='macro')
    balanced_acc = balanced_accuracy_score(labels, preds)

    return {
        'balanced_accuracy': balanced_acc,
        'accuracy': acc,
        'f1_macro': f1_macro,
        'precision': accuracy,
        'recall': recall
    }


def get_parser():
    parser = ArgumentParser()
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--dataset_name', type=str, default='ecore_555')
    parser.add_argument('--model_name', type=str, default='bert-base-uncased')
    parser.add_argument('--ckpt', type=str, default=None)
    parser.add_argument('--max_length', type=int, default=512)
    parser.add_argument('--k', type=int, default=10)

    parser.add_argument('--num_epochs', type=int, default=10)

    parser.add_argument('--warmup_steps', type=int, default=500)
    parser.add_argument('--num_log_steps', type=int, default=20)
    parser.add_argument('--num_eval_steps', type=int, default=20)
    parser.add_argument('--num_save_steps', type=int, default=20)
    parser.add_argument('--train_batch_size', type=int, default=32)
    parser.add_argument('--eval_batch_size', type=int, default=128)
    parser.add_argument('--lr', type=float, default=1e-5)

    return parser.parse_args()



if __name__ == '__main__':

    args = get_parser()

    dataset_name = args.dataset_name
    model_name = args.model_name


    texts = [
        (g['txt'], g['labels'])
        for file_name in os.listdir(f'datasets/{dataset_name}')
        for g in json.load(open(f'datasets/{dataset_name}/{file_name}'))
        if 'ecore' in file_name and file_name.endswith('.jsonl')
    ]
    shuffle(texts)
    labels = [y for _, y in texts]
    y_map = {label: i for i, label in enumerate(set(y for y in labels))}
    y = [y_map[y] for y in labels]
    n = len(texts)

    texts = [text for text, _ in texts]

    num_classes = len(y_map)

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    k = args.k
    kfold = StratifiedKFold(n_splits=k, shuffle=True, random_state=args.seed)
    all_idx = list(range(len(texts)))

    i = 0
    for train_idx, test_idx in kfold.split(np.zeros(n), np.zeros(n)):

        print(f'Fold {i+1}/{k}')

        train_texts = [texts[i] for i in train_idx]
        test_texts = [texts[i] for i in test_idx]
        train_y = [y[i] for i in train_idx]
        test_y = [y[i] for i in test_idx]


        print(f'Train: {len(train_texts)}, Test: {len(test_texts)}', num_classes)

        train_dataset = EncodingDataset(tokenizer, train_texts, train_y)
        test_dataset = EncodingDataset(tokenizer, test_texts, test_y)

        import pickle
        with open('train_dataset.pkl', 'wb') as f:
            pickle.dump(train_dataset, f)
        
        with open('test_dataset.pkl', 'wb') as f:
            pickle.dump(test_dataset, f)

        model = AutoModelForSequenceClassification.from_pretrained(args.ckpt if args.ckpt else model_name, num_labels=num_classes)
        model.resize_token_embeddings(len(tokenizer))

        print("Training model")
        output_dir = os.path.join(
            'results',
            dataset_name,
            'graph_cls',
        )

        logs_dir = os.path.join(
            'logs',
            dataset_name,
            'graph_cls',
        )

        # Training arguments
        training_args = TrainingArguments(
            output_dir=output_dir,
            num_train_epochs=args.num_epochs,
            eval_strategy="steps",
            per_device_train_batch_size=2,
            per_device_eval_batch_size=128,
            warmup_steps=500,
            weight_decay=0.01,
            logging_dir=logs_dir,
            logging_steps=10,
            eval_steps=10,
            fp16=True
        )

        # Trainer
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=test_dataset,
            compute_metrics=compute_metrics            
        )

        # Train the model
        trainer.train()
        results = trainer.evaluate()
        print(results)

        i += 1