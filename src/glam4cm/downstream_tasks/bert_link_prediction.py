from collections import Counter
import os
from transformers import TrainingArguments, Trainer
from glam4cm.data_loading.graph_dataset import GraphEdgeDataset
from glam4cm.models.hf import get_model
from glam4cm.settings import LP_TASK_LINK_PRED
from glam4cm.downstream_tasks.common_args import get_bert_args_parser, get_common_args_parser, get_config_params
from glam4cm.downstream_tasks.utils import get_models_dataset
from glam4cm.tokenization.special_tokens import *


from sklearn.metrics import (
    f1_score, 
    precision_score, 
    balanced_accuracy_score,
    recall_score
)

from glam4cm.tokenization.utils import get_tokenizer
from glam4cm.utils import merge_argument_parsers, set_seed


def compute_metrics(pred):
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    print(Counter(preds), Counter(labels))
    acc = (preds == labels).mean()
    # roc = roc_auc_score(labels, preds)
    f1_macro = f1_score(labels, preds)
    f1_micro = f1_score(labels, preds, )
    precision = precision_score(labels, preds)
    balanced_accuracy = balanced_accuracy_score(labels, preds)
    recall = recall_score(labels, preds)

    return {
        'accuracy': acc,
        'f1_macro': f1_macro,
        'f1_micro': f1_micro,
        'precision': precision,
        'recall': recall,
        'balanced_accuracy': balanced_accuracy
    }



def get_parser():

    common_parser = get_common_args_parser()
    bert_parser = get_bert_args_parser()
    parser = merge_argument_parsers(common_parser, bert_parser)

    return parser


def run(args):
    set_seed(args.seed)

    config_params = dict(
        min_enr = args.min_enr,
        min_edges = args.min_edges,
        remove_duplicates = args.remove_duplicates,
        reload = args.reload,
        language = args.language
    )
    dataset_name = args.dataset
    dataset = get_models_dataset(dataset_name, **config_params)

    print("Loaded dataset")


    graph_data_params = get_config_params(args)
    graph_data_params = {**graph_data_params, 'task': LP_TASK_LINK_PRED}

    print("Loading graph dataset")
    graph_dataset = GraphEdgeDataset(
        dataset, 
        dict(
            **graph_data_params, 
            add_negative_train_samples=args.add_negative_train_samples, 
            neg_sampling_ratio=args.neg_sampling_ratio,
            task=LP_TASK_LINK_PRED
    ))
    print("Loaded graph dataset")



    model_name = args.model_name
    tokenizer = get_tokenizer(model_name, args.use_special_tokens)


    print("Getting link prediction data")
    bert_dataset = graph_dataset.get_link_prediction_lm_data(
        tokenizer=tokenizer,
        task_type=LP_TASK_LINK_PRED
    )

    print("Training model")
    model = get_model(args.ckpt if args.ckpt else model_name, num_labels=2, len_tokenizer=len(tokenizer))

    if args.freeze_pretrained_weights:
        for param in model.base_model.parameters():
            param.requires_grad = False


    output_dir = os.path.join(
        'results',
        dataset_name,
        'lp',
        f"{graph_dataset.config_hash}",
    )

    logs_dir = os.path.join(
        'logs',
        dataset_name,
        'lp',
        f"{graph_dataset.config_hash}",
    )
    
    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=args.num_epochs,
        per_device_train_batch_size=args.train_batch_size,
        per_device_eval_batch_size=args.eval_batch_size,
        weight_decay=0.01,
        logging_dir=logs_dir,
        logging_steps=200,
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
    print(trainer.evaluate())
    trainer.save_model()
