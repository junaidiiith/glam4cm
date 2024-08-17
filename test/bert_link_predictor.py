from collections import Counter
from argparse import ArgumentParser
import os
from transformers import TrainingArguments, Trainer
import torch
from data_loading.graph_dataset import GraphDataset
from data_loading.data import ModelDataset
from encoding.common import oversample_dataset
from settings import (
    LP_TASK_LINK_PRED,
    LP_TASK_EDGE_CLS
)
from tokenization.special_tokens import *
from tokenization.utils import get_special_tokens, get_tokenizer
from transformers import BertForSequenceClassification

from sklearn.metrics import (
    roc_auc_score, 
    f1_score, 
    precision_score, 
    recall_score
)
import torch.nn.functional as F

from utils import set_seed


def compute_metrics(pred):
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    print(Counter(preds), Counter(labels))
    acc = (preds == labels).mean()
    roc = roc_auc_score(labels, preds)
    f1_macro = f1_score(labels, preds)
    f1_micro = f1_score(labels, preds, )
    precision = precision_score(labels, preds)
    recall = recall_score(labels, preds)

    return {
        'accuracy': acc,
        'roc_auc': roc,
        'f1_macro': f1_macro,
        'f1_micro': f1_micro,
        'precision': precision,
        'recall': recall
    }



def parse_args():
    parser = ArgumentParser()
    parser.add_argument('--dataset', type=str, default='ecore_555', choices=['modelset', 'ecore_555', 'mar-ecore-github'])
    parser.add_argument('--remove_duplicates', action='store_true')
    parser.add_argument('--distance', type=int, default=2)
    parser.add_argument('--model', type=str, default='bert-base-uncased')
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--reload', action='store_true')
    parser.add_argument('--timeout', type=int, default=120)
    parser.add_argument('--tr', type=float, default=0.2)
    parser.add_argument('--min_enr', type=float, default=1.2)
    parser.add_argument('--min_edges', type=int, default=10)
    parser.add_argument('--neg_sampling_ratio', type=int, default=1)
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
    task_name = LP_TASK_LINK_PRED
    distance = args.distance
    dataset = ModelDataset(dataset_name, reload=args.reload, **config_params)

    print("Loaded dataset")

    graph_data_params = dict(
        distance=distance,
        reload=args.reload,
        test_ratio=args.tr,
        add_negative_train_samples=True,
        neg_sampling_ratio=args.neg_sampling_ratio
    )

    print("Loading graph dataset")
    graph_dataset = GraphDataset(dataset, **graph_data_params)
    print("Loaded graph dataset")

    model_name = args.model
    special_tokens = get_special_tokens()
    max_length = 512
    tokenizer = get_tokenizer(model_name, special_tokens, max_length)

    print("Getting link prediction data")
    bert_dataset = graph_dataset.get_link_prediction_data(
        tokenizer=tokenizer,
        distance=distance,
        task_type=task_name
    )


    print("Training model")
    model = BertForSequenceClassification.from_pretrained(model_name, num_labels=2)
    model.resize_token_embeddings(len(tokenizer))

    output_dir = os.path.join(
        'results',
        dataset_name,
        task_name
    )

    logs_dir = os.path.join(
        'logs',
        dataset_name,
        task_name
    )

    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=32,
        per_device_eval_batch_size=128,
        warmup_steps=500,
        weight_decay=0.01,
        logging_dir=logs_dir,
        logging_steps=200,
        eval_strategy='steps',
        eval_steps=500,
        save_steps=500,
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