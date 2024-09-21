from sklearn.model_selection import train_test_split
import torch
import json
import os
from data_loading.metadata import (
    ArchimateMetaData, 
    EcoreMetaData
)
from embeddings.common import Embedder
from lang2graph.archimate import ArchiMateNxG
from lang2graph.ecore import EcoreNxG
from lang2graph.common import (
    create_graph_from_edge_index, 
    get_node_texts,
    get_edge_data
)

from tokenization.special_tokens import *
from torch_geometric.data import Data
from torch_geometric.transforms import RandomLinkSplit
import torch
from torch_geometric.data import Data, Dataset
from typing import List, Optional, Sequence, Union


def edge_index_to_idx(graph, edge_index):
    return torch.tensor(
        [
            graph.edge_to_idx[(u, v)] 
            for u, v in edge_index.t().tolist()
        ], 
        dtype=torch.long
    )


class TorchGraph:
    def __init__(
            self, 
            graph: Union[EcoreNxG, ArchiMateNxG], 
            metadata: Union[EcoreMetaData, ArchimateMetaData],
            save_dir: str,
            distance = 1,
            test_ratio=0.2,
            use_edge_types=False,
            use_attributes=False,
            use_special_tokens=False
        ):

        self.graph = graph
        self.metadata = metadata
        
        self.raw_data = graph.xmi if hasattr(graph, 'xmi') else graph.json_obj
        self.use_edge_types = use_edge_types
        self.use_attributes = use_attributes
        self.use_special_tokens = use_special_tokens
        
        self.distance = distance
        self.test_ratio = test_ratio
        self.save_dir = save_dir

    
    def get_node_edge_strings(self, data: Data, edge_index, embedder: Embedder=None):
        node_texts = self.get_graph_node_strs(
            edge_index, 
            self.distance, 
            self.use_special_tokens
        ) ### Considering nodes with edges only in the subgraph

        edge_texts = self.get_graph_edge_strs_from_node_strs(
            node_strs=node_texts, 
            edge_index=self.graph.edge_index, 
            use_edge_types=self.use_edge_types,
            use_special_tokens=self.use_special_tokens
        )

        if embedder is not None:
            node_embeddings = embedder.embed(list(node_texts.values()))
            edge_embeddings = embedder.embed(list(edge_texts.values()))

            setattr(data, 'x', node_embeddings)
            setattr(data, 'edge_attr', edge_embeddings)
        
        return node_texts, edge_texts

    

    def get_graph_node_strs(self, edge_index: torch.Tensor, distance = None, use_special_tokens=False):
        if distance is None:
            distance = self.distance

        subgraph = create_graph_from_edge_index(self.graph, edge_index)
        return get_node_texts(
            subgraph, 
            distance,
            label=self.metadata.node_label,
            use_attributes=self.use_attributes,
            attribute_labels=self.metadata.node_attributes,
            use_special_tokens=use_special_tokens
        )
    

    def get_graph_edge_strs_from_node_strs(
            self, 
            node_strs, 
            edge_index: torch.Tensor, 
            use_edge_types=False,
            neg_samples=False,
            use_special_tokens=False
        ):

        edge_strs = dict()
        label = self.metadata.edge_label
        for u, v in edge_index.t().tolist():
            u_str = node_strs[u]
            v_str = node_strs[v]
            u_label = self.graph.id_to_node_label[u]
            v_label = self.graph.id_to_node_label[v]

            if not neg_samples:
                edge_data = self.graph.edges[u_label, v_label]
                edge_label = edge_data.get(label, "")
                edge_type = get_edge_data(edge_data, 'type', self.metadata.type)
            else:
                edge_label = ""
                edge_type = ""
            
            if use_special_tokens:
                if use_edge_types:
                    edge_str = f"{NODE_LABEL} {u_str} {EDGE_START} {edge_label} {edge_type} {EDGE_END} {NODE_LABEL} {v_str}"
                else:
                    edge_str = f"{NODE_LABEL} {u_str} {EDGE_START} {edge_label} {EDGE_END} {NODE_LABEL} {v_str}"
            else:
                if use_edge_types:
                    edge_str = f"{u_str} {edge_label} {edge_type} {v_str}"
                else:
                    edge_str = f"{u_str} {edge_label} {v_str}"

            edge_strs[(u, v)] = edge_str

        # if len(edge_strs):
        #     print(list(edge_strs.values())[-1])
        return edge_strs
    

    def validate_data(self):
        assert self.data.num_nodes == self.graph.number_of_nodes()
        

    @property
    def name(self):
        return '.'.join(self.graph.graph_id.replace('/', '_').split('.')[:-1])


    def save_to_mapping(self):
        graph_embedding_file_map = dict()
        fp = f'{self.save_dir}/mapping.json'
        if os.path.exists(fp):
            graph_embedding_file_map = json.load(open(fp, 'r'))
        else:
            graph_embedding_file_map = dict()
        
        graph_embedding_file_map[self.name] = self.graph.id
        json.dump(graph_embedding_file_map, open(fp, 'w'), indent=4)



class TorchEdgeGraph(TorchGraph):
    def __init__(
            self, 
            graph: Union[EcoreNxG, ArchiMateNxG], 
            metadata: Union[EcoreMetaData, ArchimateMetaData],
            save_dir: str,
            distance = 1,
            test_ratio=0.2,

            embedder: Embedder=None,
            use_neg_samples=False,
            neg_samples_ratio=1,
            use_edge_types=False,
            use_attributes=False,
            use_special_tokens=False,
        ):

        super().__init__(
            graph, 
            metadata, 
            save_dir, 
            distance, 
            test_ratio, 
            use_edge_types, 
            use_attributes, 
            use_special_tokens=use_special_tokens,
        )
        self.use_neg_samples = use_neg_samples
        self.neg_sampling_ratio = neg_samples_ratio
        self.data, self.node_texts, self.edge_texts = self.get_pyg_data(embedder)
        self.validate_data()
    
    
    def get_pyg_data(self, embedder: Embedder):
        
        d = Data()

        transform = RandomLinkSplit(
            num_val=0, 
            num_test=self.test_ratio, 
            add_negative_train_samples=self.use_neg_samples,
            neg_sampling_ratio=self.neg_sampling_ratio,
            split_labels=True
        )

        try:
            train_data, _, test_data = transform(Data(
                edge_index=self.graph.edge_index, 
                num_nodes=self.graph.number_of_nodes()
            ))
        except IndexError as e:
            print(self.graph.edge_index)
            raise e

        train_idx = edge_index_to_idx(self.graph, train_data.edge_index)
        test_idx = edge_index_to_idx(self.graph, test_data.pos_edge_label_index)
        train_mask = torch.zeros(self.graph.edge_index.size(1), dtype=torch.bool)
        train_mask[train_idx] = True
        test_mask = torch.zeros(self.graph.edge_index.size(1), dtype=torch.bool)
        test_mask[test_idx] = True

        setattr(d, 'train_edge_mask', train_mask)
        setattr(d, 'test_edge_mask', test_mask)


        assert all([self.graph.numbered_graph.has_edge(*edge) for edge in train_data.edge_index.t().tolist()])
        assert all([self.graph.numbered_graph.has_edge(*edge) for edge in test_data.pos_edge_label_index.t().tolist()])

        if hasattr(test_data, 'neg_edge_label_index'):
            assert not any([self.graph.numbered_graph.has_edge(*edge) for edge in test_data.neg_edge_label_index.t().tolist()])
        else:
            test_data.neg_edge_label_index = torch.tensor([], dtype=torch.long)
            test_data.neg_edge_label = torch.tensor([], dtype=torch.long)


        if hasattr(train_data, 'neg_edge_label_index'):
            assert not any([self.graph.numbered_graph.has_edge(*edge) for edge in train_data.neg_edge_label_index.t().tolist()])
        else:
            train_data.neg_edge_label_index = torch.tensor([], dtype=torch.long)
            train_data.neg_edge_label = torch.tensor([], dtype=torch.long)

        setattr(d, 'train_pos_edge_label_index', train_data.pos_edge_label_index)
        setattr(d, 'train_pos_edge_label', train_data.pos_edge_label)
        setattr(d, 'train_neg_edge_label_index', train_data.neg_edge_label_index)
        setattr(d, 'train_neg_edge_label', train_data.neg_edge_label)
        setattr(d, 'test_pos_edge_label_index', test_data.pos_edge_label_index)
        setattr(d, 'test_pos_edge_label', test_data.pos_edge_label)
        setattr(d, 'test_neg_edge_label_index', test_data.neg_edge_label_index)
        setattr(d, 'test_neg_edge_label', test_data.neg_edge_label)


        edge_index = train_data.edge_index
        setattr(d, 'overall_edge_index', self.graph.edge_index)
        setattr(d, 'edge_index', edge_index)

        node_texts, edge_texts = self.get_node_edge_strings(
            d,
            edge_index=edge_index, 
            embedder=embedder
        )

        setattr(d, 'num_nodes', self.graph.number_of_nodes())
        return d, node_texts, edge_texts
    

    @property
    def save_idx(self):
        path = os.path.join(self.save_dir, f'eg_d={self.distance}_tr={self.test_ratio}_{self.graph.id}')
        return path


    # def load_pyg_data(self, embedder: Embedder=None):

    #     if os.path.exists(self.save_idx):
    #         self.save_to_mapping()
    #         self.data = torch.load(f"{self.save_idx}/data.pt")
    #         self.node_texts = pickle.load(open(f"{self.save_idx}/node_texts.pkl", 'rb'))
    #         self.edge_texts = pickle.load(open(f"{self.save_idx}/edge_texts.pkl", 'rb'))

    #         if embedder is not None and (self.data.x is None or self.regen_embeddings):
    #             node_embeddings = embedder.embed(list(self.node_texts.values()))
    #             self.data.x = node_embeddings

    #         if embedder is not None and (self.data.edge_attr is None or self.regen_embeddings):
    #             edge_embeddings = embedder.embed(list(self.edge_texts.values()))
    #             self.data.edge_attr = edge_embeddings
            
    #         self.save()
    #         return True

    #     return False


    # def save(self):
    #     os.makedirs(self.save_idx, exist_ok=True)
    #     torch.save(self.data, f"{self.save_idx}/data.pt")
    #     pickle.dump(self.node_texts, open(f"{self.save_idx}/node_texts.pkl", 'wb'))
    #     pickle.dump(self.edge_texts, open(f"{self.save_idx}/edge_texts.pkl", 'wb'))
    #     self.save_to_mapping()



class TorchNodeGraph(TorchGraph):
    def __init__(
            self, 
            graph: Union[EcoreNxG, ArchiMateNxG], 
            metadata: dict,
            save_dir: str,
            distance = 1,
            test_ratio=0.2,
            embedder: Embedder=None,
            use_edge_types=False,
            use_attributes=False,
            use_special_tokens=False,
        ):

        super().__init__(
            graph, 
            metadata, 
            save_dir, 
            distance, 
            test_ratio, 
            use_edge_types, 
            use_attributes=use_attributes,
            use_special_tokens=use_special_tokens
        )
        
        self.data, self.node_texts, self.edge_texts = self.get_pyg_data(embedder)
        self.validate_data()
            
    
    def get_pyg_data(self, embedder: Embedder):
        d = Data()
        train_nodes, test_nodes = train_test_split(
            list(self.graph.numbered_graph.nodes), test_size=self.test_ratio, shuffle=True, random_state=42
        )

        train_idx = torch.tensor(train_nodes, dtype=torch.long)
        test_idx = torch.tensor(test_nodes, dtype=torch.long)

        train_mask = torch.zeros(self.graph.number_of_nodes(), dtype=torch.bool)
        train_mask[train_idx] = True
        test_mask = torch.zeros(self.graph.number_of_nodes(), dtype=torch.bool)
        test_mask[test_idx] = True


        setattr(d, 'train_node_mask', train_mask)
        setattr(d, 'test_node_mask', test_mask)



        assert all([self.graph.numbered_graph.has_node(n) for n in train_nodes])
        assert all([self.graph.numbered_graph.has_node(n) for n in test_nodes])

        edge_index = self.graph.edge_index
        setattr(d, 'edge_index', edge_index)

        node_texts, edge_texts = self.get_node_edge_strings(
            d,
            edge_index=edge_index, 
            embedder=embedder
        )

        setattr(d, 'num_nodes', self.graph.number_of_nodes())
        return d, node_texts, edge_texts
        

    @property
    def name(self):
        return '.'.join(self.graph.graph_id.replace('/', '_').split('.')[:-1])


    @property
    def save_idx(self):
        path = os.path.join(self.save_dir, f'ng_d={self.distance}_tr={self.test_ratio}_{self.graph.id}')
        return path


    # def load_pyg_data(self, embedder=None):

    #     if os.path.exists(self.save_idx):
    #         self.save_to_mapping()
    #         self.data = torch.load(f"{self.save_idx}/data.pt")
    #         self.node_texts = pickle.load(open(f"{self.save_idx}/node_texts.pkl", 'rb'))
    #         if embedder is not None and (self.data.x is None or self.regen_embeddings):
    #             print("Embeddings not found. Generating...")
    #             node_embeddings = embedder.embed(list(self.node_texts.values()))
    #             self.data.x = node_embeddings
    #             self.save()

    #         return True

    #     return False


    # def save(self):
    #     os.makedirs(self.save_idx, exist_ok=True)
    #     torch.save(self.data, f"{self.save_idx}/data.pt")
    #     pickle.dump(self.node_texts, open(f"{self.save_idx}/node_texts.pkl", 'wb'))
    #     self.save_to_mapping()



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

        return Data(
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