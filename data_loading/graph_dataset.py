import os
from random import shuffle
from sklearn.model_selection import StratifiedKFold
import torch
import numpy as np
from transformers import AutoTokenizer
from data_loading.data import TorchGraph
from data_loading.models_dataset import ModelDataset
from data_loading.encoding import EncodingDataset
from tqdm.auto import tqdm
from embeddings.common import get_embedding_model
from settings import seed, graph_data_dir
from settings import (
    LP_TASK_EDGE_CLS,
    LP_TASK_LINK_PRED,
)



class GraphDataset(torch.utils.data.Dataset):
    def __init__(
            self, 
            models_dataset: ModelDataset,
            save_dir='datasets/graph_data',
            distance=1,
            test_ratio=0.2,
            add_negative_train_samples=False,
            neg_sampling_ratio=1,
            use_edge_types=False,
            reload=False,
            use_embeddings=False,
            embed_model_name='bert-base-uncased',
            ckpt=None,
        ):
        super().__init__()
        self.distance = distance
        self.save_dir = f'{save_dir}/{models_dataset.name}'
        embedder = get_embedding_model(embed_model_name, ckpt) if use_embeddings else None
        os.makedirs(self.save_dir, exist_ok=True)
        self._c = {label:j for j, label in enumerate({g.label for g in models_dataset})}
        for i in range(len(models_dataset)):
            models_dataset[i].label = self._c[models_dataset[i].label]

        self.graphs = [
            TorchGraph(
                g, 
                save_dir=self.save_dir,
                distance=distance,
                test_ratio=test_ratio,
                reload=reload,
                use_neg_samples=add_negative_train_samples,
                neg_samples_ratio=neg_sampling_ratio,
                use_edge_types=use_edge_types,
            ) 
            for g in tqdm(models_dataset, desc='Creating graphs')
        ]
        for g in tqdm(self.graphs, desc='Processing graphs'):
            g.process_graph(embedder)

        self._c = {label:j for j, label in enumerate({g.label for g in models_dataset})}
        self.labels = torch.tensor([self._c[g.label] for g in models_dataset], dtype=torch.long)
        self.num_classes = len(self._c)

    def __len__(self):
        return len(self.graphs)
    

    def __getitem__(self, index: int):
        return self.graphs[index]
    

    def __add__(self, other):
        self.graphs += other.graphs
        self.labels = torch.cat([self.labels, other.labels])
        self._c.update(other._c)
        self.num_classes = len(self._c)

        return self
    

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
    

    def get_link_prediction_lm_data(
            self, 
            tokenizer: AutoTokenizer,
            distance, 
            task_type=LP_TASK_EDGE_CLS
        ):
        if task_type == LP_TASK_EDGE_CLS:
            data = {'train_edges': [], 'train_edge_classes': [], 'test_edges': [], 'test_edge_classes': []}
        elif task_type == LP_TASK_LINK_PRED:
            data = {'train_pos_edges': [], 'train_neg_edges': [], 'test_pos_edges': [], 'test_neg_edges': []}
        for graph in tqdm(self.graphs, desc='Getting link prediction data'):
            pos_edge_idx = graph.data.edge_index
            node_strs = graph.get_graph_node_strs(pos_edge_idx, distance)
            train_pos_edge_index = graph.data.edge_index
            train_neg_edge_index = graph.data.train_neg_edge_label_index
            test_pos_edge_index = graph.data.test_pos_edge_label_index
            test_neg_edge_index = graph.data.test_neg_edge_label_index

            train_pos_edges = list(graph.get_graph_edge_strs_from_node_strs(node_strs, train_pos_edge_index).values())
            train_neg_edges = list(graph.get_graph_edge_strs_from_node_strs(node_strs, train_neg_edge_index).values())
            test_pos_edges = list(graph.get_graph_edge_strs_from_node_strs(node_strs, test_pos_edge_index).values())
            test_neg_edges = list(graph.get_graph_edge_strs_from_node_strs(node_strs, test_neg_edge_index).values())

            train_edge_classes = graph.data.edge_classes[graph.data.train_edge_idx] 
            test_edge_classes = graph.data.edge_classes[graph.data.test_edge_idx]

            if task_type == LP_TASK_EDGE_CLS:
                data['train_edges'] += train_pos_edges
                data['train_edge_classes'] += train_edge_classes
                data['test_edges'] += test_pos_edges
                data['test_edge_classes'] += test_edge_classes
            elif task_type == LP_TASK_LINK_PRED:
                data['train_pos_edges'] += train_pos_edges
                data['train_neg_edges'] += train_neg_edges
                data['test_pos_edges'] += test_pos_edges
                data['test_neg_edges'] += test_neg_edges
        

        print("Tokenizing data")
        if task_type == LP_TASK_EDGE_CLS:
            
            datasets = {
                'train': EncodingDataset(
                    tokenizer, 
                    data['train_edges'], 
                    data['train_edge_classes']
                ),
                'test': EncodingDataset(
                    tokenizer, 
                    data['test_edges'], 
                    data['test_edge_classes']
                )
            }
        elif task_type == LP_TASK_LINK_PRED:
            datasets = {
                'train': EncodingDataset(
                    tokenizer, 
                    data['train_pos_edges'] + data['train_neg_edges'], 
                    [1] * len(data['train_pos_edges']) + [0] * len(data['train_neg_edges'])
                ),
                'test': EncodingDataset(
                    tokenizer, 
                    data['test_pos_edges'] + data['test_neg_edges'], 
                    [1] * len(data['test_pos_edges']) + [0] * len(data['test_neg_edges'])
                )
            }
        
        print("Tokenized data")
        
        return datasets