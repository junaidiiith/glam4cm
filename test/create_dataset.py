import random
import torch
import numpy as np

from data_loading.models_dataset import ArchiMateModelDataset, EcoreModelDataset
from data_loading.graph_dataset import (
    GraphNodeDataset,
    GraphEdgeDataset
)
from test.common_args import get_common_args_parser


def get_parser():
    parser = get_common_args_parser()
    return parser.parse_args().parse_args()


def run(args):
    
    seed = 42
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

    
    config_params = dict(
        min_enr = args.min_enr,
        min_edges = args.min_edges,
    )
    ecore = EcoreModelDataset('ecore_555', reload=args.reload, **config_params)
    modelset = EcoreModelDataset('modelset', reload=args.reload, remove_duplicates=True, **config_params)
    mar = EcoreModelDataset('mar-ecore-github', reload=args.reload, **config_params)
    eamodelset = ArchiMateModelDataset('eamodelset', reload=args.reload, **config_params)

    graph_data_params = dict(
        distance=args.distance,
        add_negative_train_samples=args.add_neg_samples,
        neg_sampling_ratio=1,
    )

    GraphEdgeDataset(ecore, reload=False, **graph_data_params)
    GraphEdgeDataset(modelset, reload=True, **graph_data_params)
    GraphEdgeDataset(mar, reload=True, **graph_data_params)
    GraphEdgeDataset(eamodelset, reload=True, **graph_data_params)


    GraphNodeDataset(ecore, reload=False, **graph_data_params)
    GraphNodeDataset(modelset, reload=True, **graph_data_params)
    GraphNodeDataset(mar, reload=True, **graph_data_params)
    GraphNodeDataset(eamodelset, reload=True, **graph_data_params)