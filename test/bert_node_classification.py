from test.common_args import get_bert_args_parser, get_common_args_parser
import os
from transformers import TrainingArguments, Trainer
from data_loading.graph_dataset import GraphNodeDataset
from data_loading.utils import oversample_dataset
from test.utils import get_models_dataset
from tokenization.special_tokens import *
from tokenization.utils import get_special_tokens, get_tokenizer
from transformers import AutoModelForSequenceClassification

from sklearn.metrics import (
    accuracy_score, 
    f1_score, 
    recall_score,
    balanced_accuracy_score
)

from utils import merge_argument_parsers, set_seed



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


def get_parser():
    common_parser = get_common_args_parser()
    bert_parser = get_bert_args_parser()
    parser = merge_argument_parsers(common_parser, bert_parser)

    parser.add_argument('--oversampling_ratio', type=float, default=-1)
    parser.add_argument('--cls_label', type=str, required=True)

    return parser.parse_args()



def run(args):
    set_seed(args.seed)

    config_params = dict(
        min_enr = args.min_enr,
        min_edges = args.min_edges,
        remove_duplicates = args.remove_duplicates,
        reload=args.reload
    )
    dataset_name = args.dataset
    distance = args.distance
    dataset = get_models_dataset(dataset_name, **config_params)

    print("Loaded dataset")

    graph_data_params = dict(
        distance=args.distance,
        reload=args.reload,
        test_ratio=args.test_ratio,
        use_attributes=args.use_attributes,
        use_edge_types=args.use_edge_types,
        
        use_embeddings=args.use_embeddings,
        embed_model_name=args.embed_model_name,
        ckpt=args.ckpt,
    )


    print("Loading graph dataset")
    graph_dataset = GraphNodeDataset(dataset, **graph_data_params)
    print("Loaded graph dataset")

    assert hasattr(graph_dataset, f'num_nodes_{args.cls_label}'), f"Dataset does not have node_{args.cls_label} attribute"
    num_labels = getattr(graph_dataset, f"num_nodes_{args.cls_label}")

    model_name = args.model_name
    special_tokens = get_special_tokens()
    max_length = args.max_length
    tokenizer = get_tokenizer(model_name, special_tokens, max_length)

    print("Getting link prediction data")
    bert_dataset = graph_dataset.get_node_classification_lm_data(
        label=args.cls_label,
        tokenizer=tokenizer,
        distance=distance,
    )

    if args.oversampling_ratio != -1:
        ind_w_oversamples = oversample_dataset(bert_dataset['train'])
        bert_dataset['train'].inputs = bert_dataset['train'][ind_w_oversamples]


    model = AutoModelForSequenceClassification.from_pretrained(args.ckpt if args.ckpt else model_name, num_labels=num_labels)
    model.resize_token_embeddings(len(tokenizer))
    
    print("Training model")
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
        num_train_epochs=args.num_epochs,
        per_device_train_batch_size=args.train_batch_size,
        per_device_eval_batch_size=args.eval_batch_size,
        weight_decay=0.01,
        logging_dir=logs_dir,
        logging_steps=args.num_log_steps,
        eval_strategy='steps',
        eval_steps=args.num_eval_steps,
        save_steps=args.num_save_steps,
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