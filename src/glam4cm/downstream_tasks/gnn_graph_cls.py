import os
from glam4cm.data_loading.graph_dataset import GraphNodeDataset
from glam4cm.models.gnn_layers import GNNConv, GraphClassifer
from glam4cm.trainers.gnn_graph_classifier import GNNGraphClassificationTrainer as Trainer
from glam4cm.downstream_tasks.common_args import get_common_args_parser, get_config_params, get_gnn_args_parser
from glam4cm.utils import merge_argument_parsers, set_seed
from glam4cm.downstream_tasks.utils import get_models_dataset


def get_parser():
    common_parser = get_common_args_parser()
    gnn_parser = get_gnn_args_parser()
    parser = merge_argument_parsers(common_parser, gnn_parser)

    parser.add_argument('--cls_label', type=str, default='label')
    parser.add_argument('--global_pool', type=str, default='mean')
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

    graph_data_params = get_config_params(args)

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

    logs_dir = os.path.join(
        "logs",
        dataset_name,
        "gnn_graph_cls",
        f"{graph_dataset.config_hash}",
    )

    for datasets in graph_dataset.get_kfold_gnn_graph_classification_data():

        edge_dim = graph_dataset[0].data.edge_attr.shape[1] if args.num_heads else None

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
            edge_dim=edge_dim,
        )

        clf_input_dim = output_dim*num_heads if args.num_heads else output_dim
        classifier = GraphClassifer(
            input_dim=clf_input_dim,
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
            logs_dir=logs_dir
        )

        trainer.run()
        break