from data_loading.graph_dataset import GraphNodeDataset
from data_loading.models_dataset import EcoreModelDataset
from models.gnn_layers import GNNConv, NodeClassifer
from tokenization.special_tokens import *
from trainers.gnn_node_classifier import Trainer
from utils import merge_argument_parsers, set_seed
from test.common_args import get_common_args_parser, get_gnn_args_parser


def parse_args():
    common_parser = get_common_args_parser()
    gnn_parser = get_gnn_args_parser()
    parser = merge_argument_parsers(common_parser, gnn_parser)
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
        use_embeddings=args.use_embeddings,
        embed_model_name=args.embed_model,
        ckpt=args.ckpt
    )

    print("Loading graph dataset")
    graph_dataset = GraphNodeDataset(dataset, **graph_data_params)
    print("Loaded graph dataset")

    num_classes = graph_dataset.num_node_classes


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

    mlp_predictor = NodeClassifer(
        input_dim=output_dim,
        hidden_dim=hidden_dim,
        num_layers=num_mlp_layers, 
        num_classes=num_classes,
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

    print("Training GNN Node Classification model")
    trainer.run()