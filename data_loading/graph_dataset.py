from collections import Counter, defaultdict
import os
from random import shuffle
from typing import Union
from sklearn.model_selection import StratifiedKFold
import torch
import numpy as np
from transformers import AutoTokenizer
from data_loading.data import TorchEdgeGraph, TorchNodeGraph
from data_loading.models_dataset import ArchiMateModelDataset, EcoreModelDataset
from data_loading.encoding import EncodingDataset
from tqdm.auto import tqdm
from embeddings.common import get_embedding_model
from lang2graph.common import get_node_data, get_edge_data
from data_loading.metadata import ArchimateMetaData, EcoreMetaData
from settings import seed
from settings import (
    LP_TASK_EDGE_CLS,
    LP_TASK_LINK_PRED,
)



class GraphDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        models_dataset: Union[EcoreModelDataset, ArchiMateModelDataset],
        save_dir='datasets/graph_data',
        distance=1,
        use_attributes=False,
        use_edge_types=False,
        
        use_embeddings=False,
        embed_model_name='bert-base-uncased',
        ckpt=None,
    ):
        if isinstance(models_dataset, EcoreModelDataset):
            self.metadata = EcoreMetaData()
        elif isinstance(models_dataset, ArchiMateModelDataset):
            self.metadata = ArchimateMetaData()

        self.distance = distance
        self.save_dir = f'{save_dir}/{models_dataset.name}'
        self.embedder = get_embedding_model(embed_model_name, ckpt) if use_embeddings else None
        os.makedirs(self.save_dir, exist_ok=True)
        self.use_edge_types = use_edge_types
        self.use_attributes = use_attributes
        self.graphs = list()

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
    

    def add_cls_labels(self):
        self.add_node_labels()
        self.add_edge_labels()
        self.add_graph_labels()


    def add_node_labels(self):
        model_type = self.metadata.type
        labels = self.metadata.node_cls
        if isinstance(labels, str):
            labels = [labels]
        
        for label in labels:
            label_values = list()
            for torch_graph in self.graphs:
                values = list()
                for _, node_data in torch_graph.graph.nodes(data=True):
                    values.append(get_node_data(node_data, label, model_type))
                label_values.append(values)
        
            self.node_label_map = {l: i for i, l in enumerate(list(set(node_label for node_labels in label_values for node_label in node_labels)))}
            label_values = [
                [
                    self.node_label_map[node_label] 
                    for node_label in node_labels
                ]
                for node_labels in label_values
            ]
            
            for torch_graph, node_classes in zip(self.graphs, label_values):
                setattr(torch_graph.data, f"node_{label}", torch.tensor(node_classes))
            
            setattr(self, f"num_nodes_{label}", len(self.node_label_map))

    def add_edge_labels(self):
        model_type = self.metadata.type
        labels = self.metadata.edge_cls
        if isinstance(labels, str):
            labels = [labels]
        
        for label in labels:
            label_values = list()
            for torch_graph in self.graphs:
                values = list()
                for _, _, edge_data in torch_graph.graph.edges(data=True):
                    values.append(get_edge_data(edge_data, label, model_type))
                label_values.append(values)
        
            self.edge_label_map = {l: i for i, l in enumerate(list(set(edge_label for edge_labels in label_values for edge_label in edge_labels)))}
            label_values = [
                [
                    self.edge_label_map[edge_label] 
                    for edge_label in edge_labels
                ]
                for edge_labels in label_values
            ]
            
            for torch_graph, edge_classes in zip(self.graphs, label_values):
                setattr(torch_graph.data, f"edge_{label}", torch.tensor(edge_classes))
            
            setattr(self, f"num_edges_{label}", len(self.edge_label_map))


    def add_graph_labels(self):
        if hasattr(self.metadata, 'graph'):
            label = self.metadata.graph_cls
            if hasattr(self.graphs[0].graph, label):
                graph_labels = list()
                for torch_graph in self.graphs:
                    graph_labels.append(getattr(torch_graph.graph, label))
            
                self.graph_label_map = {l: i for i, l in enumerate(list(set(graph_label for graph_label in graph_labels)))}
                graph_labels = [
                    self.graph_label_map[graph_label]
                    for graph_label in graph_labels
                ]

                for torch_graph, graph_label in zip(self.graphs, graph_labels):
                    setattr(torch_graph.data, f"graph_{label}", torch.tensor([graph_label]))
                

class GraphEdgeDataset(GraphDataset):
    def __init__(
            self, 
            models_dataset: Union[EcoreModelDataset, ArchiMateModelDataset],
            save_dir='datasets/graph_data',
            distance=1,
            test_ratio=0.2,
            add_negative_train_samples=False,
            neg_sampling_ratio=1,
            reload=False,
            use_embeddings=False,
            use_attributes=False,
            use_edge_types=False,
            embed_model_name='bert-base-uncased',
            ckpt=None,
        ):
        super().__init__(
            models_dataset=models_dataset,
            save_dir=save_dir,
            distance=distance,
            use_embeddings=use_embeddings,
            embed_model_name=embed_model_name,
            ckpt=ckpt,
            use_edge_types=use_edge_types,
            use_attributes=use_attributes,
        )

        if use_attributes:
            assert self.metadata.node_attributes is not None, "Node attributes are not defined in metadata to be used"

        self.graphs = [
            TorchEdgeGraph(
                g, 
                save_dir=self.save_dir,
                metadata=self.metadata,
                distance=distance,
                test_ratio=test_ratio,
                reload=reload,
                use_neg_samples=add_negative_train_samples,
                neg_samples_ratio=neg_sampling_ratio,
                use_edge_types=use_edge_types,
                use_attributes=use_attributes
            ) 
            for g in tqdm(models_dataset, desc='Creating graphs')
        ]
        for g in tqdm(self.graphs, desc='Processing graphs'):
            g.process_graph(self.embedder)

        self.add_cls_labels()

        train_count, test_count = dict(), dict()
        for g in self.graphs:
            train_idx = g.data.train_edge_idx
            test_idx = g.data.test_edge_idx
            train_labels = getattr(g.data, f'edge_{self.metadata.edge_cls}')[train_idx]
            test_labels = getattr(g.data, f'edge_{self.metadata.edge_cls}')[test_idx]
            t1 = dict(Counter(train_labels.tolist()))
            t2 = dict(Counter(test_labels.tolist()))
            for k in t1:
                train_count[k] = train_count.get(k, 0) + t1[k]
            
            for k in t2:
                test_count[k] = test_count.get(k, 0) + t2[k]

        print(f"Train edge classes: {train_count}")
        print(f"Test edge classes: {test_count}")
    

    def get_link_prediction_lm_data(
            self, 
            label: str,
            tokenizer: AutoTokenizer,
            distance, 
            task_type=LP_TASK_EDGE_CLS
        ):
        data = defaultdict(list)
        for graph in tqdm(self.graphs, desc='Getting link prediction data'):
            pos_edge_idx = graph.data.edge_index
            
            node_strs = graph.get_graph_node_strs(pos_edge_idx, distance)
            train_pos_edge_index = graph.data.edge_index
            train_neg_edge_index = graph.data.train_neg_edge_label_index
            test_pos_edge_index = graph.data.test_pos_edge_label_index
            test_neg_edge_index = graph.data.test_neg_edge_label_index

            edge_indices = {
                'train_pos': train_pos_edge_index,
                'train_neg': train_neg_edge_index,
                'test_pos': test_pos_edge_index,
                'test_neg': test_neg_edge_index
            }

            for edge_index_label, edge_index in edge_indices.items():
                if "neg" in edge_index_label and task_type == LP_TASK_LINK_PRED:
                    continue
                edge_strs = graph.get_graph_edge_strs_from_node_strs(
                    node_strs, 
                    edge_index,
                    use_edge_types=self.use_edge_types,
                    neg_samples="neg" in edge_index_label
                )
                edge_strs = list(edge_strs.values())
                data[f'{edge_index_label}_edges'] += edge_strs

            train_idx = graph.data.train_edge_idx
            test_idx = graph.data.test_edge_idx
            data['train_edge_classes'] += getattr(graph.data, f'edge_{label}')[train_idx]
            data['test_edge_classes'] += getattr(graph.data, f'edge_{label}')[test_idx]


        print("Tokenizing data")
        if task_type == LP_TASK_EDGE_CLS:
            
            datasets = {
                'train': EncodingDataset(
                    tokenizer, 
                    data['train_pos_edges'], 
                    data['train_edge_classes']
                ),
                'test': EncodingDataset(
                    tokenizer, 
                    data['test_pos_edges'], 
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
    


class GraphNodeDataset(GraphDataset):
    def __init__(
            self, 
            models_dataset: Union[EcoreModelDataset, ArchiMateModelDataset],
            save_dir='datasets/graph_data',
            distance=1,
            test_ratio=0.2,
            reload=False,
            
            use_attributes=False,
            use_edge_types=False,

            use_embeddings=False,
            embed_model_name='bert-base-uncased',
            ckpt=None,
        ):
        super().__init__(
            models_dataset=models_dataset,
            save_dir=save_dir,
            distance=distance,
            use_embeddings=use_embeddings,
            use_edge_types=use_edge_types,
            embed_model_name=embed_model_name,
            ckpt=ckpt
        )

        self.graphs = [
            TorchNodeGraph(
                g, 
                metadata=self.metadata,
                save_dir=self.save_dir,
                distance=distance,
                test_ratio=test_ratio,
                reload=reload,
                use_attributes=use_attributes,
                use_edge_types=use_edge_types
            ) 
            for g in tqdm(models_dataset, desc='Creating graphs')
        ]
        for g in tqdm(self.graphs, desc='Processing graphs'):
            g.process_graph(self.embedder)

        self.add_cls_labels()
        
        node_labels = self.metadata.node_cls
        if isinstance(node_labels, str):
            node_labels = [node_labels]

        for node_label in node_labels:
            print(f"Node label: {node_label}")
            train_count, test_count = dict(), dict()
            for g in self.graphs:
                train_idx = g.data.train_node_idx
                test_idx = g.data.test_node_idx
                train_labels = getattr(g.data, f'node_{node_label}')[train_idx]
                test_labels = getattr(g.data, f'node_{node_label}')[test_idx]
                t1 = dict(Counter(train_labels.tolist()))
                t2 = dict(Counter(test_labels.tolist()))
                for k in t1:
                    train_count[k] = train_count.get(k, 0) + t1[k]
                
                for k in t2:
                    test_count[k] = test_count.get(k, 0) + t2[k]

            print(f"Train Node classes: {train_count}")
            print(f"Test Node classes: {test_count}")


    def get_node_classification_lm_data(
            self, 
            label,
            tokenizer: AutoTokenizer,
            distance, 
        ):
        data = {'train_nodes': [], 'train_node_classes': [], 'test_nodes': [], 'test_node_classes': []}
        for graph in tqdm(self.graphs, desc='Getting link prediction data'):
            node_strs = list(graph.get_graph_node_strs(graph.data.edge_index, distance).values())
            train_node_strs = [node_strs[i.item()] for i in graph.data.train_node_idx]
            test_node_strs = [node_strs[i.item()] for i in graph.data.test_node_idx]
            
            train_node_classes = getattr(graph.data, f'node_{label}')[graph.data.train_node_idx]
            test_node_classes = getattr(graph.data, f'node_{label}')[graph.data.test_node_idx]
            
            data['train_nodes'] += train_node_strs
            data['train_node_classes'] += train_node_classes.tolist()
            data['test_nodes'] += test_node_strs
            data['test_node_classes'] += test_node_classes.tolist()
                    

        print("Tokenizing data")
            
        dataset = {
            'train': EncodingDataset(
                tokenizer, 
                data['train_nodes'], 
                data['train_node_classes']
            ),
            'test': EncodingDataset(
                tokenizer, 
                data['test_nodes'], 
                data['test_node_classes']
            )
        }
        
        print("Tokenized data")
        
        return dataset