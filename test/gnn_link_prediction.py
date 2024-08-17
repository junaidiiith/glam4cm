from argparse import ArgumentParser
from data_loading.graph_dataset import GraphDataset
from data_loading.data import ModelDataset
from models.gnn_layers import GNNModel, MLPPredictor
from settings import (
    LP_TASK_LINK_PRED,
    LP_TASK_EDGE_CLS
)
from tokenization.special_tokens import *
from trainers.link_predictor import GNNLinkPredictionTrainer
from utils import randomize_features, set_seed



num_classes_map = {
    'edge_cls': 3,
    'lp': 2
}


def parse_args():
    parser = ArgumentParser()
    parser.add_argument('--dataset', type=str, default='ecore_555', choices=['modelset', 'ecore_555', 'mar-ecore-github'])
    parser.add_argument('--remove_duplicates', action='store_true')
    parser.add_argument('--distance', type=int, default=1)
    parser.add_argument('--task', type=str, default=LP_TASK_LINK_PRED, choices=[LP_TASK_EDGE_CLS, LP_TASK_LINK_PRED])
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
    parser.add_argument('--aggregation', type=str, default='sum')
    parser.add_argument('--dropout', type=float, default=0.3)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--num_epochs', type=int, default=100)
    parser.add_argument('--bias', action='store_true')
    parser.add_argument('--l_norm', action='store_true')

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
    graph_dataset = GraphDataset(dataset, **graph_data_params)
    print("Loaded graph dataset")


    randomize = args.randomize or graph_dataset[0].data.x is None
    input_dim = args.input_dim

    torch_dataset = [graph_dataset[i].data for i, _ in enumerate(graph_dataset)]

    if randomize:
        torch_dataset = randomize_features(torch_dataset, input_dim)


    task = args.task
    model_name = args.gnn_model

    num_classes = num_classes_map[task]
    hidden_dim = args.hidden_dim
    output_dim = args.output_dim
    num_conv_layers = args.num_conv_layers
    num_mlp_layers = args.num_mlp_layers
    num_heads = args.num_heads
    residual = True
    l_norm = False
    dropout = args.dropout
    randomize = args.randomize
    aggregation = args.aggregation


    gnn_conv_model = GNNModel(
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

    mlp_predictor = MLPPredictor(
        h_feats=output_dim,
        num_layers=num_mlp_layers, 
        num_classes=num_classes,
        bias=True,
    )

    trainer = GNNLinkPredictionTrainer(
        gnn_conv_model, 
        mlp_predictor, 
        torch_dataset,
        task_type=task,
        lr=1e-3,
        num_epochs=100,
        batch_size=32
    )

    print("Training GNN Link Prediction model on {} task".format(task))

    trainer.run_epochs(num_epochs=args.num_epochs)