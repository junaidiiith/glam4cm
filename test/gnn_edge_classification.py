from data_loading.graph_dataset import GraphEdgeDataset
from models.gnn_layers import GNNConv, EdgeClassifer
from test.utils import get_models_dataset
from tokenization.special_tokens import *
from trainers.gnn_edge_classifier import GNNEdgeClassificationTrainer as Trainer
from utils import set_seed, merge_argument_parsers
from test.common_args import get_common_args_parser, get_gnn_args_parser


def get_parser():
    common_parser = get_common_args_parser()
    gnn_parser = get_gnn_args_parser()
    parser = merge_argument_parsers(common_parser, gnn_parser)
    parser.add_argument('--cls_label', type=str, default='type')
    return parser.parse_args()


def run(args):

    set_seed(args.seed)
    
    config_params = dict(
        timeout = args.timeout,
        min_enr = args.min_enr,
        min_edges = args.min_edges,
        remove_duplicates = args.remove_duplicates,
        reload = args.reload,
    )
    dataset_name = args.dataset

    dataset = get_models_dataset(dataset_name, **config_params)

    graph_data_params = dict(
        reload=args.reload,
        test_ratio=args.test_ratio,
        add_negative_train_samples=True,
        neg_sampling_ratio=args.neg_sampling_ratio,
        distance=args.distance,
        use_embeddings=args.use_embeddings,
        embed_model_name=args.embed_model_name,
        ckpt=args.ckpt
    )

    print("Loading graph dataset")
    graph_dataset = GraphEdgeDataset(dataset, **graph_data_params)
    print("Loaded graph dataset")


    randomize = args.randomize or graph_dataset[0].data.x is None
    input_dim = args.input_dim


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

    num_edges_label = f"num_edges_{args.cls_label}"
    assert hasattr(graph_dataset, num_edges_label), f"Graph dataset does not have attribute {num_edges_label}"
    num_classes = getattr(graph_dataset, num_edges_label)


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
        input_dim=output_dim,
        hidden_dim=hidden_dim,
        num_layers=num_mlp_layers, 
        num_classes=num_classes,
        bias=args.bias,
    )



    trainer = Trainer(
        gnn_conv_model, 
        mlp_predictor, 
        graph_dataset.get_torch_geometric_data(),
        cls_label=args.cls_label,
        lr=args.lr,
        num_epochs=args.num_epochs,
        batch_size=args.batch_size,
        randomize_ne=randomize
    )

    print("Training GNN Edge Classification model")
    trainer.run()