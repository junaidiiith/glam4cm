import os
from glam4cm.downstream_tasks.common_args import (
    get_common_args_parser,
    get_config_params,
    get_config_str, 
    get_gpt_args_parser
)

from glam4cm.data_loading.graph_dataset import GraphNodeDataset
from glam4cm.models.cmgpt import CMGPT, CMGPTClassifier
from glam4cm.downstream_tasks.utils import get_models_dataset
from glam4cm.tokenization.utils import get_tokenizer
from glam4cm.trainers.cm_gpt_trainer import CMGPTTrainer
from glam4cm.utils import merge_argument_parsers, set_encoded_labels, set_seed
from glam4cm.settings import NODE_CLS_TASK, results_dir


def get_parser():
    common_parser = get_common_args_parser()
    bert_parser = get_gpt_args_parser()
    parser = merge_argument_parsers(common_parser, bert_parser)

    parser.add_argument('--cls_label', type=str)
    parser.add_argument('--pretr', type=str, default=None)
    return parser


def run(args):
    

    tokenizer = get_tokenizer('bert-base-cased', use_special_tokens=args.use_special_tokens)

    
    dataset_name = args.dataset
    print("Training model")
    output_dir = os.path.join(
        results_dir,
        dataset_name,
        f'LM_{NODE_CLS_TASK}',
        f'{args.node_cls_label}',
        get_config_str(args)
    )

    # if os.path.exists(output_dir):
    #     print(f"Output directory {output_dir} already exists. Exiting.")
    #     exit(0)

    config_params = dict(
        include_dummies = args.include_dummies,
        min_enr = args.min_enr,
        min_edges = args.min_edges,
        remove_duplicates = args.remove_duplicates,
        reload=args.reload,
        language = args.language
    )
    dataset_name = args.dataset
    distance = args.distance
    dataset = get_models_dataset(dataset_name, **config_params)

    print("Loaded dataset")

    graph_data_params = {**get_config_params(args), 'task_type': NODE_CLS_TASK}
    print("Loading graph dataset")
    graph_dataset = GraphNodeDataset(dataset, **graph_data_params)

    assert hasattr(graph_dataset, f'num_nodes_{args.node_cls_label}'), f"Dataset does not have node labels for {args.node_cls_label}"

    node_label_dataset = graph_dataset.get_node_classification_lm_data(
        args.node_cls_label,
        tokenizer=tokenizer,
        distance=args.distance,
    )
    
    set_encoded_labels(node_label_dataset['train'], node_label_dataset['test'])

    print("Training model")
    output_dir = os.path.join(
        results_dir,
        args.dataset,
        f'LM_{NODE_CLS_TASK}',
        f'{args.node_cls_label}',
    )

    logs_dir = os.path.join(
        'logs',
        args.dataset,
        f'CMGPT_{NODE_CLS_TASK}',
        f'{args.node_cls_label}',
        f"{graph_dataset.config_hash}",
    )


    if args.pretr and os.path.exists(args.pretr):
        print(f"Loading pretrained model from {args.pretr}")
        cmgpt = CMGPT.from_pretrained(f"{args.pretr}")
    else:
        print("Creating new model")
        cmgpt = CMGPT(
            vocab_size=len(tokenizer),
            embed_dim=args.embed_dim,
            block_size=args.block_size,
            n_layer=args.n_layer,
            n_head=args.n_head,
        )
    print(f"Train dataset size: {len(node_label_dataset['train'])}")
    print(f"Test dataset size: {len(node_label_dataset['test'])}")
    cmgpt_classifier = CMGPTClassifier(cmgpt, num_classes=getattr(graph_dataset, f"num_nodes_{args.node_cls_label}"))

    trainer = CMGPTTrainer(
        cmgpt_classifier, 
        train_dataset=node_label_dataset['train'],
        test_dataset=node_label_dataset['test'],
        batch_size=args.batch_size, 
        num_epochs=args.num_epochs,
        log_dir=logs_dir,
        results_dir=output_dir,
    )

    trainer.train()

    trainer.save_model()
    