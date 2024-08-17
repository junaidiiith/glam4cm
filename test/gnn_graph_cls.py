from data_loading.models_dataset import ModelDataset
from embeddings.bert import BertEmbedder
from data_loading.graph_dataset import GraphDataset
from models.gnn_layers import GNNModel, MLPPredictor
from trainers.graph_classifier import GNNTrainer
from argparse import ArgumentParser

from utils import randomize_features, set_seed


def parse_args():
    parser = ArgumentParser()
    parser.add_argument('--dataset', type=str, default='modelset', choices=['modelset', 'ecore', 'mar'])
    parser.add_argument('--remove_duplicates', action='store_true')
    parser.add_argument('--embed_model', type=str, default='bert-base-uncased')
    parser.add_argument('--ckpt', type=str, default='results/checkpoint-1380')
    parser.add_argument('--gnn_conv_model', type=str, default='SAGEConv')
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--reload', action='store_true')
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


    model_name = args.gnn_model

    num_classes = len([g.y for g in graph_dataset])
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


    gnn_trainer = GNNTrainer(
        gnn_conv_model,
        mlp_predictor,
        graph_dataset,
    )

    gnn_trainer.train_epochs(1)