from data_loading.models_dataset import ModelDataset
from embeddings.bert import BertEmbedder
from data_loading.graph_dataset import GraphDataset
from models.gnn_layers import GNNClassifier
from trainers.graph_classifier import GNNTrainer
from argparse import ArgumentParser


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
    reload = args.reload
    modelset = ModelDataset(
        args.dataset, 
        reload=reload, 
        remove_duplicates=args.remove_duplicates
    )

    embedder = BertEmbedder(args.embed_model, ckpt=args.ckpt)
    graph_dataset = GraphDataset(modelset, embedder)

    graph_classifier = GNNClassifier(
        gnn_conv_model=args.gnn_conv_model,
        input_dim=graph_dataset.num_features,
        hidden_dim=64,
        output_dim=graph_dataset.num_classes,
        num_layers=2,
        num_heads=None,
        dropout=0.1,
        residual=False,
        pool='sum',
        use_appnp=True,
        K=10,
        alpha=0.1
    )

    gnn_trainer = GNNTrainer(
        graph_classifier,
        graph_dataset,
    )

    gnn_trainer.train_epochs(1)