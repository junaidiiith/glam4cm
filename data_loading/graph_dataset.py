import os
from random import shuffle
from sklearn.model_selection import StratifiedKFold
import torch
import numpy as np
from data_loading.data import TorchGraph
from data_loading.models_dataset import ModelDataset
from tqdm.auto import tqdm
from settings import seed, graph_data_dir


class GraphDataset(torch.utils.data.Dataset):
    def __init__(
            self, 
            models_dataset: ModelDataset,
            save_dir='datasets/graph_data',
            distance=1,
            reload=False,
        ):
        self.save_dir = f'{save_dir}/{models_dataset.name}'
        os.makedirs(self.save_dir, exist_ok=True)
        self.graphs = [
            TorchGraph(
                g, 
                save_dir=self.save_dir,
                distance=distance,
                reload=reload,
            ) 
            for g in tqdm(models_dataset, desc=f'Processing {models_dataset.name}')
        ]

        self._c = {label:j for j, label in enumerate({g.label for g in models_dataset})}
        self.labels = torch.tensor([self._c[g.label] for g in models_dataset], dtype=torch.long)
        self.num_classes = len(self._c)
        self.num_features = self.graphs[0].data.x.shape[-1]

    def __len__(self):
        return len(self.graphs)
    
    def __getitem__(self, index: int):
        d = self.graphs[index].data
        y = self.labels[index]
        d.y = torch.tensor([y], dtype=torch.long)
        return d
    
    def get_train_test_split(self, tr=0.8):
        n = len(self.graphs)
        train_size = int(n * tr)
        idx = list(range(n))
        shuffle(idx)
        train_idx = idx[:train_size]
        test_idx = idx[train_size:]
        return train_idx, test_idx
    

    def k_fold_split(self, k=10):
        kfold = StratifiedKFold(n_splits=k, shuffle=True, random_state=seed)
        n = len(self.graphs)
        for train_idx, test_idx in kfold.split(np.zeros(n), np.zeros(n)):
            yield train_idx, test_idx



def get_model_embeddings_dataset(
        dataset: ModelDataset,
        save_dir=graph_data_dir,
        distance=1,
        reload=False,
    ):
    dataset = GraphDataset(
        dataset, 
        save_dir=save_dir,
        distance=distance,
        reload = reload,
    )
    return dataset
