from typing import List
from tqdm.auto import tqdm
import pickle
from random import shuffle
from sklearn.model_selection import StratifiedKFold
import json
import os
from settings import (
    datasets_dir, 
    seed,
)
import numpy as np
from lang2graph.uml import EcoreNxG



class ModelDataset:
    def __init__(
            self, 
            dataset_name: str, 
            dataset_dir=datasets_dir,
            save_dir='datasets/pickles',
            reload=False,
            remove_duplicates=False,
            use_type=False,
            extension='.jsonl',
            min_edges: int = -1,
            min_enr: float = -1,
            timeout=-1
        ):
        self.name = dataset_name
        self.dataset_dir = dataset_dir
        self.save_dir = save_dir
        self.extension = extension
        self.min_edges = min_edges
        self.min_enr = min_enr
        os.makedirs(save_dir, exist_ok=True)

        dataset_exists = os.path.exists(os.path.join(save_dir, f'{dataset_name}.pkl'))
        if reload or not dataset_exists:
            self.graphs: List[EcoreNxG] = []
            data_path = os.path.join(dataset_dir, dataset_name)
            for file in os.listdir(data_path):
                if file.endswith(self.extension) and file.startswith('ecore'):
                    json_objects = json.load(open(os.path.join(data_path, file)))
                    for g in tqdm(json_objects, desc=f'Loading {dataset_name.title()}'):
                        if remove_duplicates and g['is_duplicated']:
                            continue
                        try:
                            nxg = EcoreNxG(
                                g, 
                                use_type=use_type, 
                                timeout=timeout
                            )
                            self.graphs.append(nxg)
                        except Exception as e:
                            continue
            self.__filter_graphs()
            self.save()
        else:
            self.load()
        
        print(f'Loaded {self.name} with {len(self.graphs)} graphs')
        
        if remove_duplicates:
            self.remove_duplicates()

        print(f'Graphs: {len(self.graphs)}')


    def remove_duplicates(self):
        self.graphs = self.dedup()

    def dedup(self) -> List[EcoreNxG]:
        return [g for g in self.graphs if not g.is_duplicated]
    
    
    def get_train_test_split(self, train_size=0.8):
        n = len(self.graphs)
        train_size = int(n * train_size)
        idx = list(range(n))
        shuffle(idx)
        train_idx = idx[:train_size]
        test_idx = idx[train_size:]
        return train_idx, test_idx
    

    def k_fold_split(
            self,  
            k=10
        ):
        kfold = StratifiedKFold(n_splits=k, shuffle=True, random_state=seed)
        n = len(self.graphs)
        for train_idx, test_idx in kfold.split(np.zeros(n), np.zeros(n)):
            yield train_idx, test_idx


    @property
    def data(self):
        X, y = [], []
        for g in self.graphs:
            X.append(g.text)
            y.append(g.label)
        
        return X, y

    def __repr__(self):
        return f'Dataset({self.name}, graphs={len(self.graphs)})'
    
    def __getitem__(self, key):
        return self.graphs[key]
    
    def __iter__(self):
        return iter(self.graphs)
    
    def __len__(self):
        return len(self.graphs)
    
    def save(self):
        print(f'Saving {self.name} to pickle')
        with open(os.path.join(self.save_dir, f'{self.name}.pkl'), 'wb') as f:
            pickle.dump(self.graphs, f)
        print(f'Saved {self.name} to pickle')


    def __filter_graphs(self):        
        graphs = list()
        for graph in self.graphs:
            addable = True
            if self.min_edges > 0 and graph.number_of_edges() < self.min_edges:
                addable = False
            if self.min_enr > 0 and graph.enr < self.min_enr:
                addable = False
            
            if addable:
                graphs.append(graph)
        
        self.graphs = graphs



    def load(self):
        print(f'Loading {self.name} from pickle')
        with open(os.path.join(self.save_dir, f'{self.name}.pkl'), 'rb') as f:
            self.graphs = pickle.load(f)
        
        self.__filter_graphs()
        print(f'Loaded {self.name} with {len(self.graphs)} graphs')
