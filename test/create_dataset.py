from argparse import ArgumentParser
import random
import torch
import numpy as np

from data_loading.models_dataset import EcoreModelDataset
from data_loading.graph_dataset import (
    GraphNodeDataset,
    GraphEdgeDataset
)

def parse_args():
    parser = ArgumentParser()
    parser.add_argument('--dataset', type=str, default='ecore_555', choices=['modelset', 'ecore_555', 'mar-ecore-github'])
    parser.add_argument('--reload', action='store_true')
    parser.add_argument('--remove_duplicates', action='store_true')
    parser.add_argument('--timeout', type=int, default=-1)

    return parser.parse_args()


def run():
    
    seed = 42
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

    
    config_params = dict(
        timeout = 120,
        min_enr = 1.2,
        min_edges = 10
    )
    ecore = EcoreModelDataset('ecore_555', reload=False, **config_params)
    modelset = EcoreModelDataset('modelset', reload=True, remove_duplicates=True, **config_params)
    mar = EcoreModelDataset('mar-ecore-github', reload=True, **config_params)

    graph_data_params = dict(
        distance=2,
        add_negative_train_samples=True,
        neg_sampling_ratio=1,
    )

    GraphEdgeDataset(ecore, reload=False, **graph_data_params)
    GraphEdgeDataset(modelset, reload=True, **graph_data_params)
    GraphEdgeDataset(mar, reload=True, **graph_data_params)


    GraphNodeDataset(ecore, reload=False, **graph_data_params)
    GraphNodeDataset(modelset, reload=True, **graph_data_params)
    GraphNodeDataset(mar, reload=True, **graph_data_params)