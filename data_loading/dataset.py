from typing import List
import networkx as nx
from tqdm.auto import tqdm
import pickle
from random import shuffle
from sklearn.model_selection import StratifiedKFold
import json
import os
from settings import (
    datasets_dir, seed
)
import numpy as np



class GenericGraph(nx.DiGraph):
    def __init__(self, json_obj: dict, use_type=False):
        super().__init__()
        self.use_type = use_type
        self.json_obj = json_obj
        self.graph_id = json_obj.get('ids')
        self.graph_type = json_obj.get('model_type')
        self.label = json_obj.get('labels')
        self.is_duplicated = json_obj.get('is_duplicated')
        self.directed = json.loads(json_obj.get('graph')).get('directed')
        self.create_graph(json_obj)
        self.text = json_obj.get('txt')


    def create_graph(self, json_obj):
        graph = json.loads(json_obj['graph'])
        nodes = graph['nodes']
        edges = graph['links']
        for node in nodes:
            self.add_node(node['id'], **node)
        for edge in edges:
            self.add_edge(edge['source'], edge['target'], **edge)
    
    # @property
    # def text(self):
    #     txt = list()
    #     for _, d in self.nodes(data=True):
    #         etype = d.get('type', '')
    #         name = d.get('name', '')
    #         node_data = f"{name}{etype if self.use_type else ''}"
    #         txt.append(node_data)
    #     return SEP.join(txt).strip()


    def get_node_embeddings(self):
        pass
        
    def __repr__(self):
        return f'{self.json_obj}\nGraph({self.graph_id}, nodes={self.number_of_nodes()}, edges={self.number_of_edges()})'


class Dataset:
    def __init__(
            self, 
            dataset_name: str, 
            dataset_dir = datasets_dir,
            save_dir = 'datasets/pickles',
            reload=False,
            remove_duplicates=False,
            extension='.jsonl'
        ):
        self.name = dataset_name
        self.dataset_dir = dataset_dir
        self.save_dir = save_dir
        self.extension = extension
        os.makedirs(save_dir, exist_ok=True)

        dataset_exists = os.path.exists(os.path.join(save_dir, f'{dataset_name}.pkl'))
        if reload or not dataset_exists:
            self.graphs: List[GenericGraph] = []
            data_path = os.path.join(dataset_dir, dataset_name)
            for file in os.listdir(data_path):
                if file.endswith(self.extension) and file.startswith('ecore'):
                    json_objects = json.load(open(os.path.join(data_path, file)))
                    self.graphs += [
                        GenericGraph(g) for g in tqdm(
                            json_objects, desc=f'Loading {dataset_name.title()}'
                        )
                    ]
            self.save()
        
        else:
            self.load()
        
        if remove_duplicates:
            self.remove_duplicates()

        print(f'Graphs: {len(self.graphs)}')


    def remove_duplicates(self):
        self.graphs = self.dedup()

    def dedup(self) -> List[GenericGraph]:
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


    def load(self):
        print(f'Loading {self.name} from pickle')
        with open(os.path.join(self.save_dir, f'{self.name}.pkl'), 'rb') as f:
            self.graphs = pickle.load(f)
        
        print(f'Loaded {self.name} from pickle')