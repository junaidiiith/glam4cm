from argparse import ArgumentParser
import os
from transformers import TrainingArguments, Trainer
from data_loading.graph_dataset import GraphNodeDataset
from data_loading.models_dataset import EcoreModelDataset
from encoding.common import oversample_dataset
from tokenization.special_tokens import *
from tokenization.utils import get_special_tokens, get_tokenizer
from transformers import BertForSequenceClassification

from sklearn.metrics import (
    accuracy_score, 
    f1_score, 
    recall_score,
    balanced_accuracy_score
)

from utils import set_seed



def compute_metrics(pred):
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
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


def get_num_labels(dataset):
    train_labels = dataset['train'][:]['labels'].unique().tolist()
    test_labels = dataset['test'][:]['labels'].unique().tolist()
    return len(set(train_labels + test_labels))


def parse_args():
    parser = ArgumentParser()
    parser.add_argument('--dataset', type=str, default='ecore_555', choices=['modelset', 'ecore_555', 'mar-ecore-github'])
    parser.add_argument('--remove_duplicates', action='store_true')
    parser.add_argument('--distance', type=int, default=1)
    parser.add_argument('--model', type=str, default='bert-base-uncased')
    parser.add_argument('--epochs', type=int, default=3)
    parser.add_argument('--reload', action='store_true')
    parser.add_argument('--timeout', type=int, default=120)
    parser.add_argument('--tr', type=float, default=0.2)
    parser.add_argument('--min_enr', type=float, default=1.2)
    parser.add_argument('--min_edges', type=int, default=10)
    parser.add_argument('--oversampling_ratio', type=float, default=-1)
    parser.add_argument('--seed', type=int, default=42)
    return parser.parse_args()



def run(args):
    set_seed(args.seed)

    config_params = dict(
        timeout = args.timeout,
        min_enr = args.min_enr,
        min_edges = args.min_edges,
        remove_duplicates = args.remove_duplicates
    )
    dataset_name = args.dataset
    distance = args.distance
    dataset = EcoreModelDataset(dataset_name, reload=args.reload, **config_params)

    print("Loaded dataset")

    graph_data_params = dict(
        distance=distance,
        reload=args.reload,
        test_ratio=args.tr
    )

    print("Loading graph dataset")
    graph_dataset = GraphNodeDataset(dataset, **graph_data_params)
    print("Loaded graph dataset")

    model_name = args.model
    special_tokens = get_special_tokens()
    max_length = 512
    tokenizer = get_tokenizer(model_name, special_tokens, max_length)

    print("Getting link prediction data")
    bert_dataset = graph_dataset.get_node_classification_lm_data(
        tokenizer=tokenizer,
        distance=distance,
    )

    if args.oversampling_ratio != -1:
        ind_w_oversamples = oversample_dataset(bert_dataset['train'])
        bert_dataset['train'].inputs = bert_dataset['train'][ind_w_oversamples]

    print("Training model")
    num_labels = get_num_labels(bert_dataset)
    print(f'Number of labels: {num_labels}')
    model = BertForSequenceClassification.from_pretrained(model_name, num_labels=num_labels)
    model.resize_token_embeddings(len(tokenizer))

    output_dir = os.path.join(
        'results',
        dataset_name,
        'node_cls',
    )

    logs_dir = os.path.join(
        'logs',
        dataset_name,
        'node_cls',
    )

    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=32,
        per_device_eval_batch_size=128,
        warmup_steps=500,
        weight_decay=0.01,
        logging_dir=logs_dir,
        logging_steps=20,
        eval_strategy='steps',
        eval_steps=20,
        save_steps=20,
        save_total_limit=2,
        load_best_model_at_end=True,
        fp16=True,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=bert_dataset['train'],
        eval_dataset=bert_dataset['test'],
        compute_metrics=compute_metrics
    )

    trainer.train()
    results = trainer.evaluate()
    print(results)