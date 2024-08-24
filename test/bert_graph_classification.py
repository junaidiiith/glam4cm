import os
from sklearn.metrics import (
    accuracy_score, 
    balanced_accuracy_score, 
    f1_score, 
    recall_score
)
from transformers import (
    AutoModelForSequenceClassification,
    Trainer,
    TrainingArguments
)

from data_loading.graph_dataset import GraphNodeDataset
from test.common_args import get_bert_args_parser, get_common_args_parser
from test.utils import get_models_dataset
from tokenization.utils import get_special_tokens, get_tokenizer
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


def get_parser():
    common_parser = get_common_args_parser()
    bert_parser = get_bert_args_parser()
    parser = merge_argument_parsers(common_parser, bert_parser)

    parser.add_argument('--cls_label', type=str, default='label')
    parser.add_argument('--remove_duplicate_graphs', action='store_true')
    parser.add_argument('--use_special_tokens', action='store_true')
    return parser.parse_args()


def run(args):
    set_seed(args.seed)
    
    config_params = dict(
        min_enr = args.min_enr,
        min_edges = args.min_edges,
        remove_duplicates = args.remove_duplicates,
        reload = args.reload,
    )
    dataset_name = args.dataset

    dataset = get_models_dataset(dataset_name, **config_params)

    graph_data_params = dict(
        distance=args.distance,
        reload=args.reload,
        test_ratio=args.test_ratio,
        use_embeddings=args.use_embeddings,
        embed_model_name=args.embed_model_name,
        ckpt=args.ckpt,
        no_shuffle=args.no_shuffle,
        randomize_ne=args.randomize,
        random_ne_dim=args.random_ne_dim,
    )

    print("Loading graph dataset")
    graph_dataset = GraphNodeDataset(dataset, **graph_data_params)
    print("Loaded graph dataset")

    model_name = args.model_name
    special_tokens = get_special_tokens() if args.use_special_tokens else None
    
    tokenizer = get_tokenizer(model_name, special_tokens, args.max_length)
    for classification_dataset in graph_dataset.get_kfold_lm_graph_classification_data(
        tokenizer,
        remove_duplicates=args.remove_duplicate_graphs
    ):
        train_dataset = classification_dataset['train']
        test_dataset = classification_dataset['test']
        num_classes = classification_dataset['num_classes']

        print(len(train_dataset), len(test_dataset), num_classes)

        model = AutoModelForSequenceClassification.from_pretrained(
            args.ckpt if args.ckpt else model_name, 
            num_labels=num_classes
        )
        model.resize_token_embeddings(len(tokenizer))

        print("Training model")
        output_dir = os.path.join(
            'results',
            dataset_name,
            f'graph_cls_{args.min_edges}',
        )

        logs_dir = os.path.join(
            'logs',
            dataset_name,
            f'graph_cls_{args.min_edges}',
        )

        # Training arguments
        training_args = TrainingArguments(
            output_dir=output_dir,
            num_train_epochs=10,
            eval_strategy="steps",
            per_device_train_batch_size=args.train_batch_size,
            per_device_eval_batch_size=args.eval_batch_size,
            warmup_steps=500,
            weight_decay=0.01,
            learning_rate=5e-5,
            logging_dir=logs_dir,
            logging_steps=50,
            eval_steps=50,
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
        break