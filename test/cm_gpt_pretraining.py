from sklearn.model_selection import train_test_split
from test.common_args import (
    get_common_args_parser, 
    get_gpt_args_parser
)

from data_loading.graph_dataset import get_models_gpt_dataset
from models.cmgpt import CMGPT
from test.utils import get_models_dataset
from tokenization.utils import get_tokenizer
from trainers.cm_gpt_trainer import CMGPTTrainer
from utils import merge_argument_parsers, set_seed


def get_parser():
    common_parser = get_common_args_parser()
    bert_parser = get_gpt_args_parser()
    parser = merge_argument_parsers(common_parser, bert_parser)
    return parser


def run(args):
    set_seed(args.seed)

    tokenizer = get_tokenizer('bert-base-cased', special_tokens=True)

    models_dataset_params = dict(
        language='en',
    )

    graph_params = dict(
        use_special_tokens=True,
        distance=1,
    )

    models_dataset = get_models_dataset(args.dataset, **models_dataset_params)
    graph_dataset = get_models_gpt_dataset(
        models_dataset, 
        tokenizer, 
        **graph_params
    )

    train_dataset, test_dataset = train_test_split(graph_dataset, test_size=0.05)

    cmgpt = CMGPT(
        vocab_size=len(tokenizer),
        embed_dim=args.embed_dim,
        block_size=args.block_size,
        n_layer=args.n_layer,
        n_head=args.n_head,
    )

    trainer = CMGPTTrainer(
        cmgpt, 
        train_dataset=train_dataset,
        test_dataset=test_dataset,
        test_ratio=0.05, 
        batch_size=args.batch_size, 
        num_epochs=args.num_epochs
    )

    trainer.train()

    trainer.save_model()