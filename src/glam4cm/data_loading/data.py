import pickle
import numpy as np
from sklearn.model_selection import train_test_split
import torch
import networkx as nx
import os
from glam4cm.data_loading.metadata import (
    ArchimateMetaData, 
    EcoreMetaData
)
from glam4cm.embeddings.common import Embedder
from glam4cm.lang2graph.archimate import ArchiMateNxG
from glam4cm.lang2graph.ecore import EcoreNxG
from glam4cm.lang2graph.common import (
    create_graph_from_edge_index,
    get_node_texts,
    get_edge_texts
)

from scipy.sparse import csr_matrix

from glam4cm.settings import LP_TASK_EDGE_CLS, LP_TASK_LINK_PRED
from glam4cm.tokenization.special_tokens import *
from torch_geometric.transforms import RandomLinkSplit
import torch
from torch_geometric.data import Data, Dataset
from typing import List, Optional, Sequence, Union
from glam4cm.tokenization.utils import doc_tokenizer



def edge_index_to_idx(graph, edge_index):
    return torch.tensor(
        [
            graph.edge_to_idx[(u, v)] 
            for u, v in edge_index.t().tolist()
        ], 
        dtype=torch.long
    )


class GraphData(Data):    
    def __inc__(self, key, value, *args, **kwargs):
        if 'index' in key or 'node_mask' in key:
            return self.num_nodes
        elif 'edge_mask' in key:
            return self.num_edges
        else:
            return 0

    def __cat_dim__(self, key, value, *args, **kwargs):
        if 'index' in key:
            return 1
        else:
            return 0


class NumpyData:
    def __init__(self, data: dict = {}):
        self.set_data(data)
    
    def __getitem__(self, key):
        return getattr(self, key)

    def set_data(self, data: dict):
        for k, v in data.items():
            if isinstance(v, torch.Tensor):
                v = v.numpy()
            setattr(self, k, v)
    
    def __repr__(self):
        response = "NumpyData(" + ", ".join([
                f"{k}={list(v.shape)}" if isinstance(v, np.ndarray) 
                else f"{k}={v}" 
                for k, v in self.__dict__.items()
            ]) + ")"
        return response

    def to_graph_data(self):
        data = GraphData()
        for k, v in self.__dict__.items():
            if isinstance(v, np.ndarray):
                v = torch.from_numpy(v)
            elif isinstance(v, csr_matrix):
                v = torch.from_numpy(v.toarray())
            elif isinstance(v, int):
                v = torch.tensor(v, dtype=torch.long)
            
            if v.dtype == torch.float64:
                v = v.float()
            
            setattr(data, k, v)
        return data


class TorchGraph:
    def __init__(
            self, 
            graph: Union[EcoreNxG, ArchiMateNxG], 
            metadata: Union[EcoreMetaData, ArchimateMetaData],
            distance = 0,
            test_ratio=0.2,
            use_edge_types=False,
            use_node_types=False,
            use_attributes=False,
            use_edge_label=False,
            use_special_tokens=False,
            no_labels=False,
            node_cls_label=None,
            edge_cls_label='type',
            fp='test_graph.pkl'
        ):

        self.fp = fp
        self.graph = graph
        self.metadata = metadata
        
        self.raw_data = graph.xmi if hasattr(graph, 'xmi') else graph.json_obj
        self.use_edge_types = use_edge_types
        self.use_node_types = use_node_types
        self.use_attributes = use_attributes
        self.use_edge_label = use_edge_label
        self.use_special_tokens = use_special_tokens
        self.no_labels = no_labels

        self.node_cls_label = node_cls_label
        self.edge_cls_label = edge_cls_label
        
        self.distance = distance
        self.test_ratio = test_ratio
        self.data = NumpyData()
    

    def load(pkl_path):
        with open(pkl_path, 'rb') as f:
            return pickle.load(f)
    
    def save(self):
        os.makedirs(os.path.dirname(self.fp), exist_ok=True)
        with open(self.fp, 'wb') as f:
            pickle.dump(self, f)

    
    def get_node_edge_strings(self, edge_index):
        node_texts = self.get_graph_node_strs(
            edge_index=edge_index, 
            distance=self.distance
        )

        edge_texts = self.get_graph_edge_strs()
        return node_texts, edge_texts
    

    def embed(
            self, 
            embedder: Union[Embedder, None], 
            reload=False,
            randomize_ne=False,
            randomize_ee=False,
            random_embed_dim=128
        ):

        def generate_embeddings():
            if randomize_ne or embedder is None:
                print("Randomizing node embeddings")
                self.data.x = np.random.randn(self.graph.number_of_nodes(), random_embed_dim)
            else:
                self.data.x = embedder.embed(list(self.node_texts.values()))
            
            if randomize_ee or embedder is None:
                print("Randomizing edge embeddings")
                self.data.edge_attr = np.random.randn(self.graph.number_of_edges(), random_embed_dim)
            else:
                self.data.edge_attr = embedder.embed(list(self.edge_texts.values()))

        if os.path.exists(f"{self.fp}") and not reload:
            with open(f"{self.fp}", 'rb') as f:
                obj: Union[TorchEdgeGraph, TorchNodeGraph] = pickle.load(f)
            if not hasattr(obj.data, 'x') or not hasattr(obj.data, 'edge_attr'):
                generate_embeddings()
                self.save()
        else:
            if embedder is not None:
                generate_embeddings()
            else:
                self.data.x = np.random.randn(self.graph.number_of_nodes(), random_embed_dim)
                self.data.edge_attr = np.random.randn(self.graph.number_of_edges(), random_embed_dim)
                  
            self.save()
    

    def get_graph_node_strs(
            self, 
            edge_index: np.ndarray, 
            distance = None, 
            preprocessor: callable = doc_tokenizer
        ):
        if distance is None:
            distance = self.distance

        subgraph = create_graph_from_edge_index(self.graph, edge_index)
        return get_node_texts(
            subgraph, 
            distance,
            metadata=self.metadata,
            use_node_attributes=self.use_attributes,
            use_node_types=self.use_node_types,
            use_edge_types=self.use_edge_types,
            use_edge_label=self.use_edge_label,
            node_cls_label=self.node_cls_label,
            edge_cls_label=self.edge_cls_label,
            use_special_tokens=self.use_special_tokens,
            no_labels=self.no_labels,
            preprocessor=preprocessor
        )
    

    def get_graph_edge_strs(
            self, 
            edge_index: np.ndarray = None, 
            neg_samples=False,
            preprocessor: callable = doc_tokenizer
        ):
        if edge_index is None:
            edge_index = self.graph.edge_index

        edge_strs = dict()
        for u, v in edge_index.T:
            edge_str = get_edge_texts(
                self.graph.numbered_graph, 
                (u, v), 
                d=self.distance, 
                metadata=self.metadata, 
                use_node_attributes=self.use_attributes, 
                use_node_types=self.use_node_types, 
                use_edge_types=self.use_edge_types,
                use_edge_label=self.use_edge_label,
                use_special_tokens=self.use_special_tokens, 
                no_labels=self.no_labels,
                preprocessor=preprocessor,
                neg_samples=neg_samples
            )

            edge_strs[(u, v)] = edge_str

        return edge_strs
    

    def validate_data(self):
        assert self.data.num_nodes == self.graph.number_of_nodes()
    
    def set_graph_label(self):
        if self.metadata.graph_label is not None and not hasattr(self.graph, self.metadata.graph_label):  #Graph has a label
            text = doc_tokenizer("\n".join(list(self.node_texts.values())))
            # print("Text:", text)
            # print("-" * 100)
            setattr(self.graph, self.metadata.graph_label, text)
        

    @property
    def name(self):
        return '.'.join(self.graph.graph_id.replace('/', '_').split('.')[:-1])



class TorchEdgeGraph(TorchGraph):
    def __init__(
            self, 
            graph: Union[EcoreNxG, ArchiMateNxG], 
            metadata: Union[EcoreMetaData, ArchimateMetaData],
            distance = 1,
            test_ratio=0.2,
            add_negative_train_samples=False,
            neg_samples_ratio=1,
            use_edge_types=False,
            use_node_types=False,
            use_edge_label=False,
            use_attributes=False,
            use_special_tokens=False,
            node_cls_label=None,
            edge_cls_label='type',
            no_labels=False,
            fp: str = 'test_graph.pkl'
        ):

        super().__init__(
            graph=graph, 
            metadata=metadata, 
            distance=distance, 
            test_ratio=test_ratio, 
            use_node_types=use_node_types,
            use_edge_types=use_edge_types,
            use_attributes=use_attributes, 
            use_edge_label=use_edge_label,
            use_special_tokens=use_special_tokens,
            no_labels=no_labels,
            node_cls_label=node_cls_label,
            edge_cls_label=edge_cls_label,
            fp=fp
        )
        self.add_negative_train_samples = add_negative_train_samples
        self.neg_sampling_ratio = neg_samples_ratio
        self.data, self.node_texts, self.edge_texts = self.get_pyg_data()
        self.validate_data()
        self.set_graph_label()

        

    
    def get_pyg_data(self):
        
        d = GraphData()

        transform = RandomLinkSplit(
            num_val=0, 
            num_test=self.test_ratio, 
            add_negative_train_samples=self.add_negative_train_samples,
            neg_sampling_ratio=self.neg_sampling_ratio,
            split_labels=True
        )

        try:
            train_data, _, test_data = transform(GraphData(
                edge_index=torch.tensor(self.graph.edge_index), 
                num_nodes=self.graph.number_of_nodes()
            ))
        except IndexError as e:
            print(self.graph.edge_index)
            raise e

        train_idx = edge_index_to_idx(self.graph, train_data.edge_index)
        test_idx = edge_index_to_idx(self.graph, test_data.pos_edge_label_index)

        setattr(d, 'train_edge_mask', train_idx)
        setattr(d, 'test_edge_mask', test_idx)


        assert all([self.graph.numbered_graph.has_edge(*edge) for edge in train_data.edge_index.t().tolist()])
        assert all([self.graph.numbered_graph.has_edge(*edge) for edge in test_data.pos_edge_label_index.t().tolist()])

        setattr(d, 'train_pos_edge_label_index', train_data.pos_edge_label_index)
        setattr(d, 'train_pos_edge_label', train_data.pos_edge_label)
        setattr(d, 'test_pos_edge_label_index', test_data.pos_edge_label_index)
        setattr(d, 'test_pos_edge_label', test_data.pos_edge_label)


        if hasattr(train_data, 'neg_edge_label_index'):
            assert not any([self.graph.numbered_graph.has_edge(*edge) for edge in train_data.neg_edge_label_index.t().tolist()])
            assert not any([self.graph.numbered_graph.has_edge(*edge) for edge in test_data.neg_edge_label_index.t().tolist()])
            setattr(d, 'train_neg_edge_label_index', train_data.neg_edge_label_index)
            setattr(d, 'train_neg_edge_label', train_data.neg_edge_label)
            setattr(d, 'test_neg_edge_label_index', test_data.neg_edge_label_index)
            setattr(d, 'test_neg_edge_label', test_data.neg_edge_label)
        
        
        nx.set_edge_attributes(
            self.graph.numbered_graph, 
            {tuple(edge): False for edge in train_data.pos_edge_label_index.T.tolist()}, 
            'masked'
        )
        nx.set_edge_attributes(
            self.graph.numbered_graph, 
            {tuple(edge): True for edge in test_data.pos_edge_label_index.T.tolist()}, 
            'masked'
        )

        edge_index = train_data.edge_index
        # import code; code.interact(local=locals())
        setattr(d, 'overall_edge_index', self.graph.edge_index)
        setattr(d, 'edge_index', edge_index)

        node_texts, edge_texts = self.get_node_edge_strings(
            edge_index=edge_index.numpy(),
        )

        setattr(d, 'num_nodes', self.graph.number_of_nodes())
        setattr(d, 'num_edges', self.graph.number_of_edges())
        d = NumpyData(d)
        return d, node_texts, edge_texts
    

    def get_link_prediction_texts(self, label, task_type, only_texts=False):
        data = dict()
        train_pos_edge_index = self.data.edge_index
        test_pos_edge_index = self.data.test_pos_edge_label_index

        if task_type == LP_TASK_LINK_PRED:
            train_neg_edge_index = self.data.train_neg_edge_label_index
            test_neg_edge_index = self.data.test_neg_edge_label_index
        else:
            train_neg_edge_index = None
            test_neg_edge_index = None

        validate_edges(self)

        # print(train_neg_edge_index.shape)

        edge_indices = {
            'train_pos': train_pos_edge_index,
            'train_neg': train_neg_edge_index,
            'test_pos': test_pos_edge_index,
            'test_neg': test_neg_edge_index
        }

        for edge_index_label, edge_index in edge_indices.items():
            if edge_index is None:
                continue
            edge_strs = self.get_graph_edge_strs(
                edge_index=edge_index,
                neg_samples="neg" in edge_index_label,
            )
            
            edge_strs = list(edge_strs.values())
            data[f'{edge_index_label}_edges'] = edge_strs


        if task_type == LP_TASK_EDGE_CLS and not only_texts:
            train_mask = self.data.train_edge_mask
            test_mask = self.data.test_edge_mask
            train_classes, test_classes = getattr(self.data, f'edge_{label}')[train_mask], getattr(self.data, f'edge_{label}')[test_mask]
            data['train_edge_classes'] = train_classes.tolist()
            data['test_edge_classes'] = test_classes.tolist()
        
        return data
    



class TorchNodeGraph(TorchGraph):
    def __init__(
            self, 
            graph: Union[EcoreNxG, ArchiMateNxG], 
            metadata: dict,
            distance = 1,
            test_ratio=0.2,
            use_node_types=False,
            use_edge_types=False,
            use_edge_label=False,
            use_attributes=False,
            use_special_tokens=False,
            no_labels=False,
            node_cls_label=None,
            edge_cls_label='type',
            fp='test_graph.pkl'
        ):

        super().__init__(
            graph, 
            metadata=metadata, 
            distance=distance, 
            test_ratio=test_ratio, 
            use_node_types=use_node_types,
            use_edge_types=use_edge_types,
            use_edge_label=use_edge_label, 
            use_attributes=use_attributes,
            use_special_tokens=use_special_tokens,
            no_labels=no_labels,
            node_cls_label=node_cls_label,
            edge_cls_label=edge_cls_label,
            fp=fp
        )
        
        self.data, self.node_texts, self.edge_texts = self.get_pyg_data()
        self.validate_data()
        self.set_graph_label()

            
    
    def get_pyg_data(self):
        d = GraphData()
        train_nodes, test_nodes = train_test_split(
            list(self.graph.numbered_graph.nodes), 
            test_size=self.test_ratio, 
            shuffle=True, 
            random_state=42
        )

        nx.set_node_attributes(self.graph.numbered_graph, {node: False for node in train_nodes}, 'masked')
        nx.set_node_attributes(self.graph.numbered_graph, {node: True for node in test_nodes}, 'masked')

        train_idx = torch.tensor(train_nodes, dtype=torch.long)
        test_idx = torch.tensor(test_nodes, dtype=torch.long)

        setattr(d, 'train_node_mask', train_idx)
        setattr(d, 'test_node_mask', test_idx)


        assert all([self.graph.numbered_graph.has_node(n) for n in train_nodes])
        assert all([self.graph.numbered_graph.has_node(n) for n in test_nodes])



        edge_index = self.graph.edge_index
        setattr(d, 'edge_index', edge_index)

        node_texts, edge_texts = self.get_node_edge_strings(
            edge_index=edge_index,
        )

        setattr(d, 'num_nodes', self.graph.number_of_nodes())
        d = NumpyData(d)
        return d, node_texts, edge_texts
        

    @property
    def name(self):
        return '.'.join(self.graph.graph_id.replace('/', '_').split('.')[:-1])



def validate_edges(graph: Union[TorchEdgeGraph, TorchNodeGraph]):

    train_pos_edge_index = graph.data.edge_index
    test_pos_edge_index = graph.data.test_pos_edge_label_index
    train_neg_edge_index = graph.data.train_neg_edge_label_index if hasattr(graph.data, 'train_neg_edge_label_index') else None
    test_neg_edge_index = graph.data.test_neg_edge_label_index if hasattr(graph.data, 'test_neg_edge_label_index') else None

    assert set((a, b) for a, b in train_pos_edge_index.T.tolist()).issubset(set(graph.graph.numbered_graph.edges()))
    assert set((a, b) for a, b in test_pos_edge_index.T.tolist()).issubset(set(graph.graph.numbered_graph.edges()))
    assert len(set((a, b) for a, b in train_pos_edge_index.T.tolist()).intersection(set((a, b) for a, b in test_pos_edge_index.T.tolist()))) == 0

    if train_neg_edge_index is not None:
        assert len(set(graph.graph.numbered_graph.edges()).intersection(set((a, b) for a, b in train_neg_edge_index.T.tolist()))) == 0

    if test_neg_edge_index is not None:
        assert len(set(graph.graph.numbered_graph.edges()).intersection(set((a, b) for a, b in test_neg_edge_index.T.tolist()))) == 0
    
    if train_neg_edge_index is not None and test_neg_edge_index is not None:
        assert len(set((a, b) for a, b in train_neg_edge_index.T.tolist()).intersection(set((a, b) for a, b in test_neg_edge_index.T.tolist()))) == 0
    
    


class LinkPredictionCollater:
    def __init__(
            self, 
            follow_batch: Optional[List[str]] = None, 
            exclude_keys: Optional[List[str]] = None
        ):
        self.follow_batch = follow_batch
        self.exclude_keys = exclude_keys

    def __call__(self, batch: List[Data]):
        # Initialize lists to collect batched properties
        x = []
        edge_index = []
        edge_attr = []
        y = []
        overall_edge_index = []
        edge_classes = []
        train_edge_mask = []
        test_edge_mask = []
        train_pos_edge_label_index = []
        train_pos_edge_label = []
        train_neg_edge_label_index = []
        train_neg_edge_label = []
        test_pos_edge_label_index = []
        test_pos_edge_label = []
        test_neg_edge_label_index = []
        test_neg_edge_label = []
        
        # Offsets for edge indices
        node_offset = 0
        edge_offset = 0

        for data in batch:
            x.append(data.x)
            edge_index.append(data.edge_index + node_offset)
            edge_attr.append(data.edge_attr)
            y.append(data.y)
            overall_edge_index.append(data.overall_edge_index + edge_offset)
            edge_classes.append(data.edge_classes)

            train_edge_mask.append(data.train_edge_mask)
            test_edge_mask.append(data.test_edge_mask)

            train_pos_edge_label_index.append(data.train_pos_edge_label_index + node_offset)
            train_pos_edge_label.append(data.train_pos_edge_label)
            train_neg_edge_label_index.append(data.train_neg_edge_label_index + node_offset)
            train_neg_edge_label.append(data.train_neg_edge_label)

            test_pos_edge_label_index.append(data.test_pos_edge_label_index + node_offset)
            test_pos_edge_label.append(data.test_pos_edge_label)
            test_neg_edge_label_index.append(data.test_neg_edge_label_index + node_offset)
            test_neg_edge_label.append(data.test_neg_edge_label)

            node_offset += data.num_nodes
            edge_offset += data.edge_attr.size(0)

        return GraphData(
            x=torch.cat(x, dim=0),
            edge_index=torch.cat(edge_index, dim=1),
            edge_attr=torch.cat(edge_attr, dim=0),
            y=torch.tensor(y),
            overall_edge_index=torch.cat(overall_edge_index, dim=1),
            edge_classes=torch.cat(edge_classes),
            train_edge_mask=torch.cat(train_edge_mask),
            test_edge_mask=torch.cat(test_edge_mask),
            train_pos_edge_label_index=torch.cat(train_pos_edge_label_index, dim=1),
            train_pos_edge_label=torch.cat(train_pos_edge_label),
            train_neg_edge_label_index=torch.cat(train_neg_edge_label_index, dim=1),
            train_neg_edge_label=torch.cat(train_neg_edge_label),
            test_pos_edge_label_index=torch.cat(test_pos_edge_label_index, dim=1),
            test_pos_edge_label=torch.cat(test_pos_edge_label),
            test_neg_edge_label_index=torch.cat(test_neg_edge_label_index, dim=1),
            test_neg_edge_label=torch.cat(test_neg_edge_label),
            num_nodes=node_offset
        )


class LinkPredictionDataLoader(torch.utils.data.DataLoader):
    def __init__(
        self,
        dataset: Union[Dataset, Sequence[Data]],
        batch_size: int = 1,
        shuffle: bool = False,
        collate_fn=None,
        follow_batch: Optional[List[str]] = None,
        exclude_keys: Optional[List[str]] = None,
        **kwargs,
    ):
        if collate_fn is None:
            collate_fn = LinkPredictionCollater(follow_batch, exclude_keys)
        
        super().__init__(
            dataset,
            batch_size,
            shuffle,
            collate_fn=collate_fn,
            **kwargs,
        )