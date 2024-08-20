import os
from transformers import TrainingArguments, Trainer
from data_loading.graph_dataset import GraphEdgeDataset
from encoding.common import oversample_dataset
from settings import LP_TASK_EDGE_CLS
from test.common_args import get_common_args_parser
from tokenization.special_tokens import *
from tokenization.utils import get_special_tokens, get_tokenizer
from transformers import BertForSequenceClassification
from test.utils import get_models_dataset

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
    accuracy = accuracy_score(labels, preds, average='macro')
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


def get_parser():
    parser = get_common_args_parser()
    parser.add_argument('--model', type=str, default='bert-base-uncased')
    parser.add_argument('--oversampling_ratio', type=float, default=-1)
    parser.add_argument('--cls_label', type=str, default='type')
    return parser.parse_args()


def run(args):
    set_seed(args.seed)

    config_params = dict(
        timeout = args.timeout,
        min_enr = args.min_enr,
        min_edges = args.min_edges,
        remove_duplicates = args.remove_duplicates,
        reload=args.reload
    )
    dataset_name = args.dataset
    distance = args.distance
    
    print("Loaded dataset")
    dataset = get_models_dataset(dataset_name, **config_params)

    graph_data_params = dict(
        distance=distance,
        reload=args.reload,
        test_ratio=args.tr
    )

    print("Loading graph dataset")
    graph_dataset = GraphEdgeDataset(dataset, **graph_data_params)
    print("Loaded graph dataset")

    assert hasattr(graph_dataset, f'num_edges_{args.cls_label}'), f"Dataset does not have node_{args.cls_label} attribute"

    model_name = args.model
    special_tokens = get_special_tokens()
    max_length = 512
    tokenizer = get_tokenizer(model_name, special_tokens, max_length)

    print("Getting link prediction data")
    bert_dataset = graph_dataset.get_link_prediction_lm_data(
        args.cls_label,
        tokenizer=tokenizer,
        distance=distance,
        task_type=LP_TASK_EDGE_CLS
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
        'edge_cls',
    )

    logs_dir = os.path.join(
        'logs',
        dataset_name,
        'edge_cls',
    )

    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=args.num_epochs,
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