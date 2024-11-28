import os
from data_loading.graph_dataset import GraphEdgeDataset
from models.gnn_layers import GNNConv, EdgeClassifer
from test.utils import get_models_dataset
from tokenization.special_tokens import *
from trainers.gnn_link_predictor import GNNLinkPredictionTrainer as Trainer
from utils import merge_argument_parsers, set_seed
from test.common_args import get_common_args_parser, get_config_params, get_gnn_args_parser


def get_parser():
    common_parser = get_common_args_parser()
    gnn_parser = get_gnn_args_parser()
    parser = merge_argument_parsers(common_parser, gnn_parser)
    return parser


def run(args):

    set_seed(args.seed)
    
    config_params = dict(
        min_enr = args.min_enr,
        min_edges = args.min_edges,
        remove_duplicates = args.remove_duplicates,
        reload = args.reload,
        language = args.language
    )
    dataset_name = args.dataset
    dataset = get_models_dataset(dataset_name, **config_params)
    
    model_name = args.gnn_conv_model
    hidden_dim = args.hidden_dim
    output_dim = args.output_dim
    num_conv_layers = args.num_conv_layers
    num_mlp_layers = args.num_mlp_layers
    num_heads = args.num_heads
    residual = True
    l_norm = args.l_norm
    dropout = args.dropout
    aggregation = args.aggregation

    
    logs_dir = os.path.join(
        "logs",
        dataset_name,
        "gnn_lp",
        f'{args.min_edges}_att_{int(args.use_attributes)}_nt_{int(args.use_edge_types)}',
    )


    graph_data_params = get_config_params(args)
    print("Loading graph dataset")
    graph_dataset = GraphEdgeDataset(dataset, **graph_data_params)

    input_dim = graph_dataset[0].data.x.shape[1]

    edge_dim = None
    if args.use_edge_attrs:
        if args.use_embeddings:
            edge_dim = graph_dataset.embedder.embedding_dim
        else:
            edge_dim = graph_dataset[0].data.edge_attr.shape[1]
    
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
        aggregation=aggregation,
        edge_dim=edge_dim
    )

    clf_input_dim = gnn_conv_model.out_dim*num_heads if args.num_heads else output_dim
    # clf_input_dim = input_dim
    mlp_predictor = EdgeClassifer(
        input_dim=clf_input_dim,
        hidden_dim=hidden_dim,
        num_layers=num_mlp_layers, 
        num_classes=2,
        edge_dim=edge_dim,
        bias=False,
    )

    
    trainer = Trainer(
        gnn_conv_model, 
        mlp_predictor, 
        graph_dataset.get_torch_dataset(),
        lr=args.lr,
        num_epochs=args.num_epochs,
        batch_size=args.batch_size,
        use_edge_attrs=args.use_edge_attrs,
        logs_dir=logs_dir
    )


    print("Training GNN Link Prediction model")
    trainer.run()