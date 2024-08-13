from argparse import ArgumentParser
import os
from transformers import TrainingArguments, Trainer
import random
import torch
import numpy as np
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



def compute_metrics_multi_classification(pred):
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    logits = torch.tensor(pred.predictions)
    probabilites = F.softmax(logits, dim=-1).numpy()
    acc = (preds == labels).mean()
    roc = roc_auc_score(labels, probabilites, multi_class='ovr')
    f1_macro = f1_score(labels, preds, average='macro')
    f1_micro = f1_score(labels, preds, average='micro')
    precision = precision_score(labels, preds, average='macro')
    recall = recall_score(labels, preds, average='macro')

    return {
        'accuracy': acc,
        'roc_auc': roc,
        'f1_macro': f1_macro,
        'f1_micro': f1_micro,
        'precision': precision,
        'recall': recall
    }

def compute_metrics_binary_classification(pred):
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    logits = torch.tensor(pred.predictions)
    probabilites = F.softmax(logits, dim=-1).numpy()
    acc = (preds == labels).mean()
    roc = roc_auc_score(labels, probabilites, multi_class='ovr')
    f1_macro = f1_score(labels, preds, average='macro')
    f1_micro = f1_score(labels, preds, average='micro')
    precision = precision_score(labels, preds, average='macro')
    recall = recall_score(labels, preds, average='macro')

    return {
        'accuracy': acc,
        'roc_auc': roc,
        'f1_macro': f1_macro,
        'f1_micro': f1_micro,
        'precision': precision,
        'recall': recall
    }


def get_num_labels(dataset):
    train_labels = dataset['train'].inputs['labels'].unique().tolist()
    test_labels = dataset['test'].inputs['labels'].unique().tolist()
    return len(set(train_labels + test_labels))


def parse_args():
    parser = ArgumentParser()
    parser.add_argument('--dataset', type=str, default='modelset', choices=['modelset', 'ecore', 'mar'])
    parser.add_argument('--model', type=str, default='bert-base-uncased')
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--reload', action='store_true')
    return parser.parse_args()



def run(args):
    seed = 42
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

    config_params = dict(
        timeout = 120,
        min_enr = 1.2,
        min_edges = 10
    )
    dataset_name = 'mar-ecore-github'
    task_name = LP_TASK_LINK_PRED
    distance = 2
    compute_metrics = compute_metrics_multi_classification \
        if task_name == LP_TASK_EDGE_CLS \
        else compute_metrics_binary_classification
    # dataset = ModelDataset('ecore_555', reload=False, **config_params)
    # dataset = ModelDataset('modelset', reload=True, remove_duplicates=True, **config_params)
    dataset = ModelDataset('mar-ecore-github', reload=False, **config_params)

    print("Loaded dataset")

    graph_data_params = dict(
        distance=distance,
        reload=False,
        add_negative_train_samples=True,
        neg_sampling_ratio=1,
        use_edge_types=False,
    )

    print("Loading graph dataset")
    graph_dataset = GraphDataset(dataset, **graph_data_params)
    print("Loaded graph dataset")

    model_name = 'bert-base-uncased'
    special_tokens = get_special_tokens()
    max_length = 512
    tokenizer = get_tokenizer(model_name, special_tokens, max_length)

    print("Getting link prediction data")
    bert_dataset = graph_dataset.get_link_prediction_data(
        tokenizer=tokenizer,
        distance=distance,
        task_type=task_name
    )

    if task_name == LP_TASK_EDGE_CLS:
        ind_w_oversamples = oversample_dataset(bert_dataset['train'])
        bert_dataset['train'].inputs = bert_dataset['train'][ind_w_oversamples]

    print("Training model")
    num_labels = get_num_labels(dataset)
    model = BertForSequenceClassification.from_pretrained(model_name, num_labels=num_labels)
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
        num_train_epochs=3,
        per_device_train_batch_size=32,
        per_device_eval_batch_size=128,
        warmup_steps=500,
        weight_decay=0.01,
        logging_dir=logs_dir,
        logging_steps=1000,
        eval_strategy='steps',
        eval_steps=200,
        save_steps=200,
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