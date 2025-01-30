import os
from glam4cm.data_loading.graph_dataset import GraphNodeDataset
from glam4cm.models.gnn_layers import GNNConv, NodeClassifier
from glam4cm.downstream_tasks.utils import get_models_dataset
from glam4cm.tokenization.special_tokens import *
from glam4cm.trainers.gnn_node_classifier import GNNNodeClassificationTrainer as Trainer
from glam4cm.utils import merge_argument_parsers, set_seed
from glam4cm.downstream_tasks.common_args import get_common_args_parser, get_config_params, get_gnn_args_parser


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
    graph_data_params = get_config_params(args)

    print("Loading graph dataset")
    graph_dataset = GraphNodeDataset(dataset, **graph_data_params)
    print("Loaded graph dataset")

    graph_torch_data = graph_dataset.get_torch_dataset()

    num_nodes_label = f"num_nodes_{args.node_cls_label}"
    assert hasattr(graph_dataset, num_nodes_label), f"Graph dataset does not have attribute {num_nodes_label}"
    num_classes = getattr(graph_dataset, f"num_nodes_{args.node_cls_label}")


    input_dim = graph_torch_data[0].x.shape[1]

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
    mlp_predictor = NodeClassifier(
        input_dim=clf_input_dim,
        hidden_dim=hidden_dim,
        num_layers=num_mlp_layers, 
        num_classes=num_classes,
        bias=True,
    )

    logs_dir = os.path.join(
        "logs",
        dataset_name,
        "gnn_node_cls",
        f"{graph_dataset.config_hash}",
    )

    trainer = Trainer(
        gnn_conv_model, 
        mlp_predictor, 
        graph_torch_data,
        cls_label=args.node_cls_label,
        exclude_labels=getattr(graph_dataset, f"node_exclude_{args.node_cls_label}"),
        lr=args.lr,
        num_epochs=args.num_epochs,
        use_edge_attrs=args.use_edge_attrs,
        logs_dir=logs_dir
    )

    print("Training GNN Node Classification model")
    trainer.run()