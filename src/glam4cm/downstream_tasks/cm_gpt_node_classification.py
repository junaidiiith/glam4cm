import os
from glam4cm.downstream_tasks.common_args import (
    get_common_args_parser, 
    get_gpt_args_parser
)

from glam4cm.data_loading.graph_dataset import GraphNodeDataset
from glam4cm.models.cmgpt import CMGPT, CMGPTClassifier
from glam4cm.downstream_tasks.utils import get_models_dataset
from glam4cm.tokenization.utils import get_tokenizer
from glam4cm.trainers.cm_gpt_trainer import CMGPTTrainer
from glam4cm.utils import merge_argument_parsers, set_seed


def get_parser():
    common_parser = get_common_args_parser()
    bert_parser = get_gpt_args_parser()
    parser = merge_argument_parsers(common_parser, bert_parser)

    parser.add_argument('--cls_label', type=str)
    parser.add_argument('--pretr', type=str, default=None)
    return parser


def run(args):
    set_seed(args.seed)

    tokenizer = get_tokenizer('bert-base-cased', special_tokens=True)

    models_dataset_params = dict(
        language='en',
    )

    graph_params = dict(
        use_special_tokens=args.use_special_tokens,
        distance=args.distance,
        reload = args.reload
    )

    models_dataset = get_models_dataset(args.dataset, **models_dataset_params)
    graph_dataset = GraphNodeDataset(models_dataset, **graph_params)

    assert hasattr(graph_dataset, f'num_nodes_{args.node_cls_label}'), f"Dataset does not have node labels for {args.node_cls_label}"

    node_label_dataset = graph_dataset.get_node_classification_lm_data(
        args.node_cls_label,
        tokenizer=tokenizer,
        distance=1,
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
    
    cmgpt_classifier = CMGPTClassifier(cmgpt, num_classes=getattr(graph_dataset, f"num_nodes_{args.node_cls_label}"))

    trainer = CMGPTTrainer(
        cmgpt_classifier, 
        train_dataset=node_label_dataset['train'],
        test_dataset=node_label_dataset['test'],
        batch_size=args.batch_size, 
        num_epochs=args.num_epochs
    )

    trainer.train()

    trainer.save_model()