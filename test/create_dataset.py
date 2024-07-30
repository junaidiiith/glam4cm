import random
import torch
import numpy as np


from argparse import ArgumentParser

def parse_args():
    parser = ArgumentParser()
    parser.add_argument('--reload', action='store_true')
    parser.add_argument('--remove_duplicates', action='store_true')
    parser.add_argument('--timeout', type=int, default=-1)

    return parser.parse_args()


def run(args):
    
    seed = 42
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)



    from data_loading.models_dataset import ModelDataset
    from data_loading.graph_dataset import GraphDataset

    
    config_params = dict(
        timeout = 120,
        min_enr = 1.2,
        min_edges = 10
    )
    ecore = ModelDataset('ecore_555', reload=False, **config_params)
    modelset = ModelDataset('modelset', reload=True, remove_duplicates=True, **config_params)
    mar = ModelDataset('mar-ecore-github', reload=True, **config_params)

    graph_data_params = dict(
        distance=2,
        reload=True
    )

    GraphDataset(ecore, **graph_data_params)
    GraphDataset(modelset, **graph_data_params)
    GraphDataset(mar, **graph_data_params)