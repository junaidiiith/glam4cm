import os
from sklearn.metrics import (
    accuracy_score, 
    balanced_accuracy_score, 
    f1_score, 
    recall_score
)
from transformers import (
    Trainer,
    TrainingArguments
)

from glam4cm.data_loading.graph_dataset import GraphNodeDataset
from glam4cm.models.hf import get_model
from glam4cm.downstream_tasks.common_args import get_bert_args_parser, get_common_args_parser, get_config_params
from glam4cm.downstream_tasks.utils import get_models_dataset
from glam4cm.tokenization.utils import get_tokenizer
from glam4cm.utils import merge_argument_parsers, set_seed



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


def get_parser():
    common_parser = get_common_args_parser()
    bert_parser = get_bert_args_parser()
    parser = merge_argument_parsers(common_parser, bert_parser)

    parser.add_argument('--cls_label', type=str, default='label')
    parser.add_argument('--remove_duplicate_graphs', action='store_true')
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
    graph_data_params = get_config_params(args)
    print("Loading graph dataset")
    graph_dataset = GraphNodeDataset(dataset, **graph_data_params)
    print("Loaded graph dataset")

    model_name = args.model_name
    tokenizer = get_tokenizer(model_name, args.use_special_tokens)

    fold_id = 0
    for classification_dataset in graph_dataset.get_kfold_lm_graph_classification_data(
        tokenizer,
        remove_duplicates=args.remove_duplicate_graphs
    ):
        train_dataset = classification_dataset['train']
        test_dataset = classification_dataset['test']
        num_labels = classification_dataset['num_classes']

        print(len(train_dataset), len(test_dataset), num_labels)

        print("Training model")
        output_dir = os.path.join(
            'results',
            dataset_name,
            f'graph_cls_',
            f"{graph_dataset.config_hash}",
        )

        logs_dir = os.path.join(
            'logs',
            dataset_name,
            f'graph_cls_',
            f"{graph_dataset.config_hash}"
        )

        model = get_model(args.ckpt if args.ckpt else model_name, num_labels, len(tokenizer))

        if args.freeze_pretrained_weights:
            for param in model.base_model.parameters():
                param.requires_grad = False

        # Training arguments
        training_args = TrainingArguments(
            output_dir=output_dir,
            num_train_epochs=args.num_epochs,
            eval_strategy="steps",
            per_device_train_batch_size=args.train_batch_size,
            per_device_eval_batch_size=args.eval_batch_size,
            warmup_steps=200,
            weight_decay=0.01,
            learning_rate=5e-5,
            logging_dir=logs_dir,
            logging_steps=args.num_log_steps,
            eval_steps=args.num_eval_steps,
            save_steps=args.num_save_steps,
            save_total_limit=2,
            load_best_model_at_end=True,
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
        
        fold_id += 1
        break