from data_loading.models_dataset import EcoreModelDataset
from data_loading.graph_dataset import GraphEdgeDataset
from models.gnn_layers import GNNConv, GraphClassifer
from trainers.gnn_graph_classifier import Trainer
from argparse import ArgumentParser

from utils import set_seed


def parse_args():
    parser = ArgumentParser()
    parser.add_argument('--seed', type=int, default=42)
    
    parser.add_argument('--dataset', type=str, default='ecore_555', choices=['modelset', 'ecore_555', 'mar-ecore-github'])
    parser.add_argument('--remove_duplicates', action='store_true')
    parser.add_argument('--timeout', type=int, default=120)
    parser.add_argument('--min_enr', type=int, default=1.2)
    parser.add_argument('--min_edges', type=int, default=10)
    parser.add_argument('--distance', type=int, default=1)
    parser.add_argument('--tr', type=float, default=0.2)
    parser.add_argument('--neg_sampling_ratio', type=float, default=1.0)
    parser.add_argument('--reload', action='store_true')
    

    parser.add_argument('--randomize', action='store_true')

    parser.add_argument('--gnn_conv_model', type=str, default='SAGEConv')
    parser.add_argument('--hidden_dim', type=int, default=128)
    parser.add_argument('--output_dim', type=int, default=128)
    parser.add_argument('--num_conv_layers', type=int, default=3)
    parser.add_argument('--num_heads', type=int, default=4)
    parser.add_argument('--dropout', type=float, default=0.3)
    parser.add_argument('--aggregation', type=str, default='sum')
    parser.add_argument('--global_pool', type=str, default='mean')
    parser.add_argument('--lr', type=float, default=1e-3)


    parser.add_argument('--use_embeddings', action='store_true')
    parser.add_argument('--embed_model', type=str, default='bert-base-uncased')
    parser.add_argument('--ckpt', type=str, default='results/checkpoint-1380')

    
    
    parser.add_argument('--epochs', type=int, default=200)
    parser.add_argument('--batch_size', type=int, default=32)
    
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

    dataset = EcoreModelDataset(dataset_name, reload=False, **config_params)

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
    input_dim = graph_dataset[0].data.x.size(1)

    model_name = args.gnn_conv_model

    num_classes = graph_dataset.num_classes
    hidden_dim = args.hidden_dim
    output_dim = args.output_dim
    num_conv_layers = args.num_conv_layers
    num_heads = args.num_heads
    residual = True
    l_norm = False
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

    classifier = GraphClassifer(
        input_dim=gnn_conv_model.out_dim,
        num_classes=num_classes,
        global_pool=args.global_pool,
    )

    trainer = Trainer(
        gnn_conv_model,
        classifier, 
        graph_dataset,
        tr=args.tr,
        lr=args.lr,
        num_epochs=args.epochs,
        batch_size=args.batch_size,
        randomize_ne=randomize
    )

    trainer.run()