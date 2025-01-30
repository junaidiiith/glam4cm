from argparse import ArgumentParser
from glam4cm.settings import (
    BERT_MODEL,
    WORD2VEC_MODEL,
    TFIDF_MODEL
)


def get_config_params(args):
    common_params = dict(
        distance=args.distance,
        reload=args.reload,
        test_ratio=args.test_ratio,

        use_attributes=args.use_attributes,
        use_node_types=args.use_node_types,
        use_edge_types=args.use_edge_types,
        use_edge_label=args.use_edge_label,
        no_labels=args.no_labels,

        use_special_tokens=args.use_special_tokens,

        use_embeddings=args.use_embeddings,
        embed_model_name=args.embed_model_name,
        ckpt=args.ckpt,
        
        no_shuffle=args.no_shuffle,
        randomize_ne=args.randomize_ne,
        randomize_ee=args.randomize_ee,
        random_embed_dim=args.random_embed_dim,

        limit = args.limit,

        node_cls_label=args.node_cls_label,
        edge_cls_label=args.edge_cls_label,
    )
    

    return common_params


def get_common_args_parser():
    parser = ArgumentParser()

    parser.add_argument('--seed', type=int, default=42)

    ### Models Dataset Creation
    parser.add_argument(
        '--dataset', 
        type=str, 
        default='ecore_555', 
        choices=[
            'modelset', 
            'ecore_555', 
            'mar-ecore-github', 
            'eamodelset',
            'ontouml'
        ]
    )
    parser.add_argument('--remove_duplicates', action='store_true')
    parser.add_argument('--reload', action='store_true')
    parser.add_argument('--min_enr', type=float, default=-1.0)
    parser.add_argument('--min_edges', type=int, default=-1)
    parser.add_argument('--language', type=str, default='en')
    
    parser.add_argument('--use_attributes', action='store_true')
    parser.add_argument('--use_edge_label', action='store_true')
    parser.add_argument('--use_edge_types', action='store_true')
    parser.add_argument('--use_node_types', action='store_true')
    parser.add_argument('--use_special_tokens', action='store_true')
    parser.add_argument('--no_labels', action='store_true')

    parser.add_argument('--node_cls_label', type=str, default=None)
    parser.add_argument('--edge_cls_label', type=str, default=None)


    parser.add_argument('--limit', type=int, default=-1)


    ### Model Dataset Loading
    parser.add_argument('--distance', type=int, default=0)
    parser.add_argument('--use_embeddings', action='store_true')
    parser.add_argument('--regen_embeddings', action='store_true')
    parser.add_argument(
        '--embed_model_name', 
        type=str, 
        default='bert-base-uncased', 
        choices=[BERT_MODEL, WORD2VEC_MODEL, TFIDF_MODEL]
    )
    parser.add_argument('--max_length', type=int, default=512)
    parser.add_argument('--ckpt', type=str, default=None)
    

    parser.add_argument('--no_shuffle', action='store_true')
    parser.add_argument('--randomize_ne', action='store_true')
    parser.add_argument('--randomize_ee', action='store_true')
    parser.add_argument('--random_embed_dim', type=int, default=128)


    parser.add_argument('--test_ratio', type=float, default=0.2)
    parser.add_argument('--add_negative_train_samples', action='store_true')
    parser.add_argument('--neg_sampling_ratio', type=int, default=1)

    parser.add_argument('--num_epochs', type=int, default=100)
    parser.add_argument('--batch_size', type=int, default=32)

    return parser


def get_gnn_args_parser():
    parser = ArgumentParser()
    parser.add_argument('--num_conv_layers', type=int, default=3)
    parser.add_argument('--num_mlp_layers', type=int, default=3)
    parser.add_argument('--num_heads', type=int, default=None)

    parser.add_argument('--input_dim', type=int, default=768)
    parser.add_argument('--hidden_dim', type=int, default=128)
    parser.add_argument('--output_dim', type=int, default=128)

    parser.add_argument('--residual', action='store_true')
    parser.add_argument('--bias', action='store_true')
    parser.add_argument('--l_norm', action='store_true')
    parser.add_argument('--aggregation', type=str, default='sum')
    parser.add_argument('--dropout', type=float, default=0.3)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--gnn_conv_model', type=str, default='SAGEConv')

    parser.add_argument('--use_edge_attrs', action='store_true')
    
    return parser


def get_bert_args_parser():
    parser = ArgumentParser()

    parser.add_argument('--freeze_pretrained_weights', action='store_true')
    parser.add_argument('--model_name', type=str, default='bert-base-uncased')

    parser.add_argument('--warmup_steps', type=int, default=200)
    parser.add_argument('--num_log_steps', type=int, default=200)
    parser.add_argument('--num_eval_steps', type=int, default=200)
    parser.add_argument('--num_save_steps', type=int, default=200)
    parser.add_argument('--train_batch_size', type=int, default=2)
    parser.add_argument('--eval_batch_size', type=int, default=128)
    return parser


def get_gpt_args_parser():
    parser = ArgumentParser()
    parser.add_argument('--model_name', type=str, default='gpt2')
    parser.add_argument('--use_special_tokens', action='store_true')

    parser.add_argument('--warmup_steps', type=int, default=200)
    parser.add_argument('--blocks', type=int, default=6)
    parser.add_argument('--block_size', type=int, default=8)
    parser.add_argument('--n_head', type=int, default=8)
    parser.add_argument('--embed_dim', type=int, default=768)
    parser.add_argument('--n_layer', type=int, default=6)
    parser.add_argument('--lr', type=float, default=1e-5)
    return parser