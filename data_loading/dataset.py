from typing import List
import torch
from tqdm.auto import tqdm
import pickle
from random import shuffle
from sklearn.model_selection import StratifiedKFold
import json
import os
from embeddings.common import Embedder
from lang2graph.common import LangGraph
from settings import (
    datasets_dir, 
    seed,
    graph_data_dir
)
import numpy as np
from lang2graph.uml import EcoreNxG
from torch_geometric.data import Data
from torch_geometric.utils import (
    negative_sampling
)
import torch_geometric.transforms as T





class ModelDataset:
    def __init__(
            self, 
            dataset_name: str, 
            dataset_dir = datasets_dir,
            save_dir = 'datasets/pickles',
            reload=False,
            remove_duplicates=False,
            use_type=False,
            remove_generic_nodes=False,
            extension='.jsonl',
            min_edges = 1
        ):
        self.name = dataset_name
        self.dataset_dir = dataset_dir
        self.save_dir = save_dir
        self.extension = extension
        os.makedirs(save_dir, exist_ok=True)

        dataset_exists = os.path.exists(os.path.join(save_dir, f'{dataset_name}.pkl'))
        if reload or not dataset_exists:
            self.graphs: List[EcoreNxG] = []
            data_path = os.path.join(dataset_dir, dataset_name)
            for file in os.listdir(data_path):
                if file.endswith(self.extension) and file.startswith('ecore'):
                    json_objects = json.load(open(os.path.join(data_path, file)))
                    for g in tqdm(json_objects, desc=f'Loading {dataset_name.title()}'):
                        nxg = EcoreNxG(
                            g, 
                            use_type=use_type, 
                            remove_generic_nodes=remove_generic_nodes
                        )
                        if nxg.number_of_edges() >= min_edges:
                            self.graphs.append(nxg)
                    
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


    def load(self):
        print(f'Loading {self.name} from pickle')
        with open(os.path.join(self.save_dir, f'{self.name}.pkl'), 'rb') as f:
            self.graphs = pickle.load(f)
        
        print(f'Loaded {self.name} from pickle')



# Create your dataset
class EncodingDataset(torch.utils.data.Dataset):
    def __init__(self, tokenizer, texts, labels=None, max_length=512):
        self.inputs = tokenizer(
            texts, 
            return_tensors='pt', 
            truncation=True, 
            padding='max_length', 
            max_length=max_length
        )
        if labels:
            self.inputs['labels'] = torch.tensor(labels, dtype=torch.long)
 

    def __len__(self):
        return len(self.inputs['input_ids'])
    

    def __getitem__(self, index):
        item = {key: val[index] for key, val in self.inputs.items()}
        return item
    

class TorchGraph:
    def __init__(
            self, 
            graph: LangGraph, 
            embedder: Embedder,
            save_dir: str,
            lp_graphs=False,
            lptr=0.2
        ):
        self.graph = graph
        self.embedder = embedder
        
        self.save_dir = save_dir
        self.lp_graphs = lp_graphs
        self.lptr = lptr if lp_graphs else None
        self.process_graph()
    

    def process_graph(self):
        if not self.load_embeddings():
            texts = self.graph.get_node_texts()
            self.embeddings = self.embedder.embed(texts)
            self.edge_index = torch.tensor(
                list(self.graph.edges), dtype=torch.long).t().contiguous()

            try:
                assert self.embeddings.shape[0] == len(self.graph.nodes)
                assert self.edge_index.shape[1] == len(self.graph.edges)
            except AssertionError as e:
                print(f'Error in {self.graph.graph_id}')
                raise e

        if self.lp_graphs:
            if not self.load_lp_graphs():
                self.add_lp_graphs()

        self.save()
    
    
    def add_lp_graphs(self):
        self.lp_graphs = get_pos_neg_graphs(
            X=self.embeddings, 
            E = self.edge_index, 
            tr=self.lptr
        )


    @property
    def name(self):
        return '.'.join(self.graph.graph_id.replace('/', '_').split('.')[:-1])


    @property
    def save_idx(self):
        path = os.path.join(self.save_dir, f'{self.graph.id}')
        if self.embedder.finetuned:
            path = f'{path}_finetuned'
        return path


    def save_to_mapping(self):
        graph_embedding_file_map = dict()
        fp = f'{self.save_dir}/mapping.json'
        if os.path.exists(fp):
            graph_embedding_file_map = json.load(open(fp, 'r'))
        else:
            graph_embedding_file_map = dict()
        
        graph_embedding_file_map[self.name] = self.graph.id
        json.dump(graph_embedding_file_map, open(fp, 'w'), indent=4)


    def load_embeddings(self):

        if os.path.exists(self.save_idx):
            self.save_to_mapping()
            self.embeddings = torch.load(f"{self.save_idx}/embeddings.pt")
            self.edge_index = torch.load(f"{self.save_idx}/edge_index.pt")
            return True

        return False

    
    def load_lp_graphs(self):
        self.lp_graphs = dict()
        lp_path = f'{self.save_idx}/lp_graphs'
        if os.path.exists(lp_path):
            for k in os.listdir(lp_path):
                self.lp_graphs[k] = torch.load(os.path.join(lp_path, k))    
            return True
        
        return False
    

    def save(self):
        os.makedirs(self.save_idx, exist_ok=True)
        torch.save(self.embeddings, f"{self.save_idx}/embeddings.pt")
        torch.save(self.edge_index, f"{self.save_idx}/edge_index.pt")

        if self.lp_graphs:
            lp_path = f'{self.save_idx}/lp_graphs'
            os.makedirs(lp_path, exist_ok=True)
            for k, v in self.lp_graphs.items():
                torch.save(v, os.path.join(lp_path, k))

        self.save_to_mapping()


class GraphDataset(torch.utils.data.Dataset):
    def __init__(
            self, 
            models_dataset: ModelDataset,
            embedder: Embedder,
            lp_graphs=False,
            save_dir='datasets/graph_data',
        ):
        self.save_dir = f'{save_dir}/{models_dataset.name}'
        os.makedirs(self.save_dir, exist_ok=True)
        self.graphs = [
            TorchGraph(
                g, 
                embedder, 
                lp_graphs=lp_graphs,
                save_dir=self.save_dir
            ) 
            for g in tqdm(models_dataset, desc=f'Processing {models_dataset.name}')
        ]

        self._c = {label:j for j, label in enumerate({g.label for g in models_dataset})}
        self.labels = torch.tensor([self._c[g.label] for g in models_dataset], dtype=torch.long)
        self.num_classes = len(self._c)
        self.num_features = self.graphs[0].embeddings.shape[-1]

    def __len__(self):
        return len(self.graphs)
    
    def __getitem__(self, index: int):
        return Data(
            x=self.graphs[index].embeddings,
            edge_index=self.graphs[index].edge_index,
            y=self.labels[index]
        )
    
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


def get_pos_neg_graphs(X, E, test_ratio=0.2):
    # Create a Data object
    data = Data(x=X, edge_index=E)

    # Apply RandomLinkSplit
    transform = T.RandomLinkSplit(
        num_val=0, 
        num_test=test_ratio, 
        is_undirected=True, 
        add_negative_train_samples=False
    )
    train_data, _, test_data = transform(data)

    # Positive edges
    train_pos_edge_index = train_data.edge_index
    test_pos_edge_index = test_data.edge_index

    # Negative edges
    train_neg_edge_index = negative_sampling(
        edge_index=train_pos_edge_index, 
        num_nodes=data.num_nodes, 
        num_neg_samples=train_pos_edge_index.size(1)
    )

    test_neg_edge_index = negative_sampling(
        edge_index=test_pos_edge_index, 
        num_nodes=data.num_nodes, 
        num_neg_samples=test_pos_edge_index.size(1)
    )

    # Create the graph objects
    train_pos_g = Data(x=X, edge_index=train_pos_edge_index)
    train_neg_g = Data(x=X, edge_index=train_neg_edge_index)
    test_pos_g = Data(x=X, edge_index=test_pos_edge_index)
    test_neg_g = Data(x=X, edge_index=test_neg_edge_index)

    graphs = {
        'train_pos_g': train_pos_g,
        'train_neg_g': train_neg_g,
        'test_pos_g': test_pos_g,
        'test_neg_g': test_neg_g,
        'train_g': train_data
    }
    return graphs



def get_embedding_graph_dataset(
        dataset: ModelDataset,
        embedder: Embedder,
        save_dir=graph_data_dir,
    ):
    dataset = GraphDataset(dataset, embedder, save_dir=save_dir)
    return dataset


def get_dataset(
        dataset_name, 
        reload=False, 
        remove_duplicates=False, 
        use_type=False, 
        remove_generic_nodes=True
    ):
    return ModelDataset(
        dataset_name, 
        reload=reload, 
        remove_duplicates=remove_duplicates, 
        use_type=use_type, 
        remove_generic_nodes=remove_generic_nodes
    )
    