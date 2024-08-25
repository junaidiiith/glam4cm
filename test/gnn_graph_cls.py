from data_loading.graph_dataset import GraphNodeDataset
from models.gnn_layers import GNNConv, GraphClassifer
from trainers.gnn_graph_classifier import GNNGraphClassificationTrainer as Trainer
from test.common_args import get_common_args_parser, get_gnn_args_parser
from utils import merge_argument_parsers, set_seed
from test.utils import get_models_dataset


def get_parser():
    common_parser = get_common_args_parser()
    gnn_parser = get_gnn_args_parser()
    parser = merge_argument_parsers(common_parser, gnn_parser)

    parser.add_argument('--cls_label', type=str, default='label')
    parser.add_argument('--global_pool', type=str, default='mean')
    return parser.parse_args()


def run(args):
    set_seed(args.seed)
    
    config_params = dict(
        min_enr = args.min_enr,
        min_edges = args.min_edges,
        remove_duplicates = args.remove_duplicates,
        reload = args.reload,
    )
    dataset_name = args.dataset

    dataset = get_models_dataset(dataset_name, **config_params)

    graph_data_params = dict(
        distance=args.distance,
        reload=args.reload,
        test_ratio=args.test_ratio,
        use_embeddings=args.use_embeddings,
        embed_model_name=args.embed_model_name,
        ckpt=args.ckpt,
        regen_embeddings=args.regen_embeddings,
        no_shuffle=args.no_shuffle,
        randomize_ne=args.randomize,
        random_ne_dim=args.random_ne_dim,
    )

    print("Loading graph dataset")
    graph_dataset = GraphNodeDataset(dataset, **graph_data_params)
    print("Loaded graph dataset")

    cls_label = f"num_graph_{args.cls_label}"
    assert hasattr(graph_dataset, cls_label), f"Dataset does not have attribute {cls_label}"
    num_classes = getattr(graph_dataset, cls_label)

    print(f"Number of classes: {num_classes}")

    model_name = args.gnn_conv_model
    hidden_dim = args.hidden_dim
    output_dim = args.output_dim
    num_conv_layers = args.num_conv_layers
    num_heads = args.num_heads
    residual = True
    l_norm = False
    dropout = args.dropout
    aggregation = args.aggregation

    input_dim = graph_dataset[0].data.x.shape[1]

    for datasets in graph_dataset.get_kfold_gnn_graph_classification_data():
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
            datasets,
            lr=args.lr,
            num_epochs=args.num_epochs,
            batch_size=args.batch_size,
            use_edge_attrs=args.use_edge_attrs,
        )

        trainer.run()
        break