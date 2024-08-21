from collections import Counter, defaultdict
import os
from random import shuffle
from typing import List, Union
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
from tokenization.utils import doc_tokenizer
from utils import randomize_features



class GraphDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        models_dataset: Union[EcoreModelDataset, ArchiMateModelDataset],
        save_dir='datasets/graph_data',
        distance=1,
        use_attributes=False,
        use_edge_types=False,
        
        test_ratio=0.2,

        use_embeddings=False,
        embed_model_name='bert-base-uncased',
        ckpt=None,

        no_shuffle=False,
        randomize_ne=False,
        random_ne_dim=768,

        tokenizer_special_tokens=None
    ):
        if isinstance(models_dataset, EcoreModelDataset):
            self.metadata = EcoreMetaData()
        elif isinstance(models_dataset, ArchiMateModelDataset):
            self.metadata = ArchimateMetaData()

        self.distance = distance
        self.save_dir = os.path.join(save_dir, models_dataset.name)
        self.embedder = get_embedding_model(embed_model_name, ckpt) if use_embeddings else None

        if not self.embedder:
            self.tokenizer = AutoTokenizer.from_pretrained(embed_model_name)
        else:
            self.tokenizer = self.embedder.tokenizer
        
        if tokenizer_special_tokens:
            self.tokenizer.add_special_tokens(tokenizer_special_tokens)

        os.makedirs(self.save_dir, exist_ok=True)
        self.use_edge_types = use_edge_types
        self.use_attributes = use_attributes
        self.graphs: List[TorchNodeGraph, TorchEdgeGraph] = []

        self.test_ratio = test_ratio

        self.no_shuffle = no_shuffle
        self.randomize_ne = randomize_ne
        self.random_ne_dim = random_ne_dim
    

    def process_graphs(self):
        for g in tqdm(self.graphs, desc='Processing graphs'):
            g.process_graph(self.embedder)

        self.add_cls_labels()
        if not self.no_shuffle:
            shuffle(self.graphs)
        
        if self.randomize_ne or self.graphs[0].data.x is None:
            randomize_features([g.data for g in self.graphs], self.random_ne_dim)


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
    

    def get_train_test_split(self):
        n = len(self.graphs)
        train_size = int(n * (1 - self.test_ratio))
        idx = list(range(n))
        shuffle(idx)
        train_idx = idx[:train_size]
        test_idx = idx[train_size:]
        return train_idx, test_idx
    

    def k_fold_split(self):
        k = int(1 / self.test_ratio)
        kfold = StratifiedKFold(n_splits=k, shuffle=True, random_state=seed)
        n = len(self.graphs)
        for train_idx, test_idx in kfold.split(np.zeros(n), np.zeros(n)):
            yield train_idx, test_idx
    

    def add_cls_labels(self):
        self.add_node_labels()
        self.add_edge_labels()
        self.add_graph_labels()
        self.add_graph_text()


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
        
            node_label_map = {l: i for i, l in enumerate(list(set(node_label for node_labels in label_values for node_label in node_labels)))}
            label_values = [
                [
                    node_label_map[node_label] 
                    for node_label in node_labels
                ]
                for node_labels in label_values
            ]
            
            for torch_graph, node_classes in zip(self.graphs, label_values):
                setattr(torch_graph.data, f"node_{label}", torch.tensor(node_classes))
            
            setattr(self, f"num_nodes_{label}", len(node_label_map))

            setattr(self, f"node_label_map_{label}", node_label_map)
            setattr(self, f"node_label_map_{label}_inv", {v: k for k, v in node_label_map.items()})


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
        
            edge_label_map = {l: i for i, l in enumerate(list(set(edge_label for edge_labels in label_values for edge_label in edge_labels)))}
            label_values = [
                [
                    edge_label_map[edge_label] 
                    for edge_label in edge_labels
                ]
                for edge_labels in label_values
            ]
            
            for torch_graph, edge_classes in zip(self.graphs, label_values):
                setattr(torch_graph.data, f"edge_{label}", torch.tensor(edge_classes))
            
            setattr(self, f"num_edges_{label}", len(edge_label_map))

            setattr(self, f"edge_label_map_{label}", edge_label_map)
            setattr(self, f"edge_label_map_{label}_inv", {v: k for k, v in edge_label_map.items()})


    def add_graph_labels(self):
        if hasattr(self.metadata, 'graph'):
            label = self.metadata.graph_cls
            if label and hasattr(self.graphs[0].graph, label):
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
                setattr(self, f"num_graph_{label}", len(self.graph_label_map))

                setattr(self, f"graph_label_map_{label}", self.graph_label_map)
                setattr(self, f"graph_label_map_{label}_inv", {v: k for k, v in self.graph_label_map.items()})



    def add_graph_text(self):
        self.graph_texts = list()
        label = self.metadata.graph_label
        for torch_graph in self.graphs:
            if label and hasattr(torch_graph.graph, label):
                self.graph_texts.append(getattr(torch_graph.graph, label))
            else:
                node_label = self.metadata.node_label
                node_names = [node_data.get(node_label, "") for _, node_data in torch_graph.graph.nodes(data=True)]
                self.graph_texts.append(doc_tokenizer(" ".join(node_names)))


    def get_torch_geometric_data(self):
        return [g.data for g in self.graphs]
    
    
    def get_gnn_graph_classification_data(self):
        train_idx, test_idx = self.get_train_test_split()
        train_data = [self.graphs[i].data for i in train_idx]
        test_data = [self.graphs[i].data for i in test_idx]
        return {
            'train': train_data,
            'test': test_data
        }


    def get_kfold_gnn_graph_classification_data(self, k=10):
        for train_idx, test_idx in self.k_fold_split(k):
            train_data = [self.graphs[i].data for i in train_idx]
            test_data = [self.graphs[i].data for i in test_idx]
            yield {
                'train': train_data,
                'test': test_data
            }


    def __get_lm_data(self, indices, tokenizer=None):
        graph_label_name = self.metadata.graph_cls
        assert graph_label_name is not None, "No Graph Label found in data. Please define graph label in metadata"
        X = [self.graph_texts[i] for i in indices]
        y = [getattr(self.graphs[i].data, f'graph_{graph_label_name}')[0].item() for i in indices]
        

        if tokenizer is None:
            assert self.tokenizer is not None, "Tokenizer is not defined. Please define an tokenizer to tokenize data"
            tokenizer = self.tokenizer

        dataset = EncodingDataset(tokenizer, X, y)
        return dataset


    def get_lm_graph_classification_data(self, tokenizer=None):
        train_idx, test_idx = self.get_train_test_split()
        train_dataset = self.__get_lm_data(train_idx, tokenizer)
        test_dataset = self.__get_lm_data(test_idx, tokenizer)

        return {
            'train': train_dataset,
            'test': test_dataset
        }
        
    
    def get_kfold_lm_graph_classification_data(self, tokenizer=None):
        k = int(1 / self.test_ratio)
        for train_idx, test_idx in self.k_fold_split(k):
            train_dataset = self.__get_lm_data(train_idx, tokenizer)
            test_dataset = self.__get_lm_data(test_idx, tokenizer)
            yield {
                'train': train_dataset,
                'test': test_dataset
            }


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
            no_shuffle=False,
            randomize_ne=False,
            random_ne_dim=768,
            tokenizer_special_tokens=None
        ):
        super().__init__(
            models_dataset=models_dataset,
            save_dir=save_dir,
            distance=distance,
            test_ratio=test_ratio,
            use_embeddings=use_embeddings,
            embed_model_name=embed_model_name,
            ckpt=ckpt,
            use_edge_types=use_edge_types,
            use_attributes=use_attributes,
            no_shuffle=no_shuffle,
            randomize_ne=randomize_ne,
            random_ne_dim=random_ne_dim,
            tokenizer_special_tokens=tokenizer_special_tokens
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

        self.process_graphs()

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

            no_shuffle=False,
            randomize_ne=False,
            random_ne_dim=768,

            tokenizer_special_tokens=None
        ):
        super().__init__(
            models_dataset=models_dataset,
            save_dir=save_dir,
            distance=distance,
            test_ratio=test_ratio,
            use_embeddings=use_embeddings,
            use_edge_types=use_edge_types,
            embed_model_name=embed_model_name,
            ckpt=ckpt,
            no_shuffle=no_shuffle,
            randomize_ne=randomize_ne,
            random_ne_dim=random_ne_dim,
            tokenizer_special_tokens=tokenizer_special_tokens
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
        self.process_graphs()
        
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