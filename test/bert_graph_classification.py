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
from tokenization.utils import get_special_tokens
from utils import merge_argument_parsers



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


    return parser.parse_args()


def run(args):

    config_params = dict(
        timeout = args.timeout,
        min_enr = args.min_enr,
        min_edges = args.min_edges,
        remove_duplicates = args.remove_duplicates,
        reload=args.reload
    )
    dataset_name = args.dataset
    dataset = get_models_dataset(dataset_name, **config_params)

    model_name = args.model_name

    print("Loaded dataset")

    graph_data_params = dict(
        distance=args.distance,
        reload=args.reload,
        test_ratio=args.test_ratio,
        use_attributes=args.use_attributes,
        use_edge_types=args.use_edge_types,

        embed_model_name=args.embed_model_name,
        ckpt=args.ckpt,
        tokenizer_special_tokens=get_special_tokens(),
    )


    print("Loading graph dataset")
    graph_dataset = GraphNodeDataset(dataset, **graph_data_params)
    print("Loaded graph dataset")



    classification_dataset = graph_dataset.get_lm_graph_classification_data()

    cls_label = f"num_graph_{args.cls_label}"
    assert hasattr(graph_dataset, cls_label), f"Dataset does not have attribute {cls_label}"
    num_classes = getattr(graph_dataset, cls_label)


    model = AutoModelForSequenceClassification.from_pretrained(args.ckpt if args.ckpt else model_name, num_labels=num_classes)
    model.resize_token_embeddings(len(graph_dataset.tokenizer))
    
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
        warmup_steps=args.warmup_steps,
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
        train_dataset=classification_dataset['train'],
        eval_dataset=classification_dataset['test'],
        compute_metrics=compute_metrics
    )

    trainer.train()
    results = trainer.evaluate()
    print(results)