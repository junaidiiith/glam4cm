from argparse import ArgumentParser
from data_loading.graph_dataset import GraphEdgeDataset
from data_loading.models_dataset import ModelDataset
from models.gnn_layers import GNNConv, EdgeClassifer
from tokenization.special_tokens import *
from trainers.gnn_edge_classifier import Trainer
from utils import randomize_features, set_seed


def parse_args():
    parser = ArgumentParser()
    parser.add_argument('--dataset', type=str, default='ecore_555', choices=['modelset', 'ecore_555', 'mar-ecore-github'])
    parser.add_argument('--remove_duplicates', action='store_true')
    parser.add_argument('--distance', type=int, default=1)
    parser.add_argument('--embed_model', type=str, default='bert-base-uncased')
    parser.add_argument('--ckpt', type=str, default='modelset_ec_ft')
    parser.add_argument('--reload', action='store_true')
    parser.add_argument('--use_embeddings', action='store_true')
    parser.add_argument('--timeout', type=int, default=120)
    parser.add_argument('--min_enr', type=float, default=1.2)
    parser.add_argument('--min_edges', type=int, default=10)
    parser.add_argument('--seed', type=int, default=42)

    parser.add_argument('--tr', type=float, default=0.2)
    parser.add_argument('--neg_sampling_ratio', type=int, default=1)

    parser.add_argument('--num_conv_layers', type=int, default=3)
    parser.add_argument('--num_mlp_layers', type=int, default=3)
    parser.add_argument('--num_heads', type=int, default=4)

    parser.add_argument('--input_dim', type=int, default=768)
    parser.add_argument('--hidden_dim', type=int, default=128)
    parser.add_argument('--output_dim', type=int, default=128)

    parser.add_argument('--residual', action='store_true')
    parser.add_argument('--randomize', action='store_true')
    parser.add_argument('--bias', action='store_true')
    parser.add_argument('--l_norm', action='store_true')
    parser.add_argument('--aggregation', type=str, default='sum')
    parser.add_argument('--dropout', type=float, default=0.3)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--num_epochs', type=int, default=100)

    parser.add_argument('--gnn_model', type=str, default='SAGEConv')

    return parser.parse_args()




def run(args):

    set_seed(args.seed)
    
    config_params = dict(
        timeout = args.timeout,
        min_enr = args.min_enr,
        min_edges = args.min_edges,
        remove_duplicates = args.remove_duplicates
    )
    dataset_name = args.dataset

    dataset = ModelDataset(dataset_name, reload=False, **config_params)

    graph_data_params = dict(
        distance=args.distance,
        reload=args.reload,
        test_ratio=args.tr,
        add_negative_train_samples=True,
        neg_sampling_ratio=args.neg_sampling_ratio,
        use_embeddings=args.use_embeddings,
        embed_model_name=args.embed_model,
        ckpt=args.ckpt
    )

    print("Loading graph dataset")
    graph_dataset = GraphEdgeDataset(dataset, **graph_data_params)
    print("Loaded graph dataset")


    randomize = args.randomize or graph_dataset[0].data.x is None
    input_dim = args.input_dim


    model_name = args.gnn_model

    hidden_dim = args.hidden_dim
    output_dim = args.output_dim
    num_conv_layers = args.num_conv_layers
    num_mlp_layers = args.num_mlp_layers
    num_heads = args.num_heads
    residual = True
    l_norm = args.l_norm
    dropout = args.dropout
    aggregation = args.aggregation


    gnn_conv_model = GNNConv(
        model_name=model_name,
        input_dim=input_dim,
        hidden_dim=hidden_dim,
        out_dim=output_dim,
        num_layers=num_conv_layers,
        num_heads=num_heads,
        residual=residual,
        l_norm=l_norm,
        dropout=dropout,
        aggregation=aggregation
    )

    mlp_predictor = EdgeClassifer(
        h_feats=output_dim,
        num_layers=num_mlp_layers, 
        num_classes=3,
        bias=True,
    )


    trainer = Trainer(
        gnn_conv_model, 
        mlp_predictor, 
        graph_dataset,
        lr=args.lr,
        num_epochs=args.num_epochs,
        batch_size=args.batch_size,
        randomize_ne=randomize
    )

    print("Training GNN Edge Classification model")
    trainer.run_epochs(num_epochs=args.num_epochs)