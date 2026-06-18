from glam4cm.data_loading.graph_dataset import GraphEdgeDataset
from glam4cm.models.gnn_layers import GNNConv, EdgeClassifer
from glam4cm.settings import LINK_PRED_TASK
from glam4cm.data_loading.models_dataset import get_models_dataset
from glam4cm.tokenization.special_tokens import *
from glam4cm.trainers.gnn_link_predictor import GNNLinkPredictionTrainer as Trainer
from glam4cm.utils import merge_argument_parsers, set_seed
from glam4cm.downstream_tasks.common_args import (
    get_common_args_parser,
    get_config_hash, 
    get_config_params, 
    get_config_str,
    get_gnn_args_parser,
    set_embed_model
)
from glam4cm.downstream_tasks.utils import get_experiment_dir, save_experiment_results
from glam4cm.diagnostics.gnn_link_prediction import write_link_prediction_diagnostics

 
def get_parser():
    common_parser = get_common_args_parser()
    gnn_parser = get_gnn_args_parser()
    parser = merge_argument_parsers(common_parser, gnn_parser)
    parser.add_argument(
        "--diagnose_gnn_lp",
        action="store_true",
        help="Build the GNN link prediction dataset, write diagnostics, and exit before training.",
    )
    parser.add_argument(
        "--diagnostics_output",
        type=str,
        default=None,
        help="Optional JSON path for --diagnose_gnn_lp output.",
    )
    parser.add_argument(
        "--lp_message_passing_edges",
        choices=["train_graph", "label_edges", "label_edges_with_negatives"],
        default="train_graph",
        help=(
            "Edges used to compute node embeddings for link prediction. "
            "train_graph uses only observed positive training edges."
        ),
    )
    return parser


def run(args):
    set_seed(args.seed)
    
    
    config_params = dict(
        include_dummies = args.include_dummies,
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

    graph_data_params = get_config_params(args)
    graph_data_params = {
        **graph_data_params, 
        'add_negative_train_samples': True, 
        'neg_sampling_ratio': args.neg_sampling_ratio,
        'task_type': LINK_PRED_TASK
    }
    
    set_embed_model(args)
    output_dir = get_experiment_dir(
        args,
        f"GNN_{LINK_PRED_TASK}",
        "link",
        get_config_str(args),

    )
    
    print("Loading graph dataset")
    graph_dataset = GraphEdgeDataset(
        dataset,
        **graph_data_params, 
    )
    set_seed(args.seed)
    
    if args.diagnose_gnn_lp:
        write_link_prediction_diagnostics(
            graph_dataset,
            args=args,
            output_path=args.diagnostics_output,
        )
        return

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
    mlp_predictor = EdgeClassifer(
        input_dim=clf_input_dim,
        hidden_dim=hidden_dim,
        num_layers=num_mlp_layers, 
        num_classes=2,
        edge_dim=edge_dim,
        bias=False,
    )

    graph_torch_data = graph_dataset.get_torch_dataset()
    # exclude_labels = getattr(graph_dataset, f"node_exclude_{args.node_cls_label}")
    # set_torch_encoding_labels(graph_torch_data, f"node_{args.node_cls_label}", exclude_labels)
    
    trainer = Trainer(
        model=gnn_conv_model, 
        predictor=mlp_predictor, 
        dataset=graph_torch_data,
        lr=args.lr,
        num_epochs=args.num_epochs,
        batch_size=args.batch_size,
        use_edge_attrs=args.use_edge_attrs,
        message_passing_edges=args.lp_message_passing_edges,
        logs_dir=output_dir
    )


    print("Training GNN Link Prediction model")
    trainer.run()
    save_experiment_results(output_dir, trainer.results)
