import pickle
from langchain_text_splitters import RecursiveCharacterTextSplitter
from sklearn.preprocessing import LabelEncoder
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
from data_loading.encoding import EncodingDataset, GPTTextDataset
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


def exclude_labels_from_data(X, labels, exclude_labels):
    X = [X[i] for i in range(len(X)) if labels[i] not in exclude_labels]
    labels = [labels[i] for i in range(len(labels)) if labels[i] not in exclude_labels]
    return X, labels


def validate_classes(graphs, label, exclude_labels, element):
    

    train_labels, test_labels = list(), list()
    train_idx = f"train_{element}_mask"
    test_idx = f"test_{element}_mask"

    for torch_graph in graphs:
        labels = getattr(torch_graph.data, f"{element}_{label}")
        mask = ~torch.isin(labels, torch.tensor(exclude_labels))
        train_mask = getattr(torch_graph.data, train_idx)
        test_mask = getattr(torch_graph.data, test_idx)
        merged_train_mask = mask & train_mask
        merged_test_mask = mask & test_mask
        train_labels.append(labels[merged_train_mask])
        test_labels.append(labels[merged_test_mask])


    train_classes = set(sum([
        getattr(torch_graph.data, f"{element}_{label}")[getattr(torch_graph.data, train_idx)].tolist() 
        for torch_graph in graphs], []
    ))
    test_classes = set(sum([
        getattr(torch_graph.data, f"{element}_{label}")[getattr(torch_graph.data, test_idx)].tolist() 
        for torch_graph in graphs], []
    ))
    num_train_classes = len(train_classes)
    num_test_classes = len(test_classes)
    print("Train classes:", train_classes)
    print("Test classes:", test_classes)
    print(f"Number of classes in training set: {num_train_classes}")
    print(f"Number of classes in test set: {num_test_classes}")

    assert num_train_classes == num_test_classes, f"Number of classes in training and test set should be the same for {label}"



class GraphDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        models_dataset: Union[EcoreModelDataset, ArchiMateModelDataset],
        save_dir='datasets/graph_data',
        distance=1,
        use_attributes=False,
        use_edge_types=False,
        use_node_types=False,
        
        test_ratio=0.2,

        use_embeddings=False,
        use_special_tokens=False,
        embed_model_name='bert-base-uncased',
        ckpt=None,

        no_shuffle=False,
        random_embed_dim=768,
        randomize_ne = False,
        randomize_ee = False,
        tokenizer_special_tokens=None,
        exclude_labels: list = [None]
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
        self.use_node_types = use_node_types
        self.use_attributes = use_attributes
        self.graphs: List[TorchNodeGraph, TorchEdgeGraph] = []

        self.test_ratio = test_ratio

        self.no_shuffle = no_shuffle
        self.random_embed_dim = random_embed_dim
        self.exclude_labels = exclude_labels

        self.randomize_ne = randomize_ne
        self.randomize_ee = randomize_ee

        self.use_special_tokens = use_special_tokens
    

    def post_process_graphs(self):
        self.add_cls_labels()
        if not self.no_shuffle:
            shuffle(self.graphs)
        
        if self.graphs[0].data.x is None or self.randomize_ne:
            print("Randomizing node embeddings")
            randomize_features([g.data for g in self.graphs], self.random_embed_dim, 'node')
        
        if self.graphs[0].data.edge_attr is None or self.randomize_ee:
            print("Randomizing edge embeddings")
            randomize_features([g.data for g in self.graphs], self.random_embed_dim, 'edge')


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


    def save(self, save_path):
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        with open(save_path, 'wb') as f:
            pickle.dump(self.graphs, f)


    def load(self, save_path):
        with open(save_path, 'rb') as f:
            self.graphs = pickle.load(f)


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
                for _, node in torch_graph.graph.nodes(data=True):
                    node_data = get_node_data(node, label, model_type)
                    values.append(node_data)
                
                label_values.append(values)
            
            node_label_map = LabelEncoder()
            node_label_map.fit_transform([j for i in label_values for j in i])
            label_values = [node_label_map.transform(i) for i in label_values]
            print(node_label_map.classes_)
            
            for torch_graph, node_classes in zip(self.graphs, label_values):
                setattr(torch_graph.data, f"node_{label}", torch.tensor(node_classes))
            
            setattr(self, f"node_label_map_{label}", node_label_map)
            
            exclude_labels = [
                node_label_map.transform([e])[0] 
                for e in self.exclude_labels
                if e in node_label_map.classes_
            ]
            setattr(self, f"node_exclude_{label}", exclude_labels)
            
            num_labels = len(node_label_map.classes_) - len(exclude_labels)

            print("Setting num_nodes_", label, num_labels)
            setattr(self, f"num_nodes_{label}", num_labels)

            if hasattr(self.graphs[0].data, 'train_node_mask'):
                validate_classes(self.graphs, label, exclude_labels, 'node')


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

            edge_label_map = LabelEncoder()
            edge_label_map.fit_transform([j for i in label_values for j in i])
            label_values = [edge_label_map.transform(i) for i in label_values]
            print("Edge Classes: ", edge_label_map.classes_)

            for torch_graph, edge_classes in zip(self.graphs, label_values):
                setattr(torch_graph.data, f"edge_{label}", torch.tensor(edge_classes))
            
            setattr(self, f"edge_label_map_{label}", edge_label_map)

            exclude_labels = [
                edge_label_map.transform([e])[0] 
                for e in self.exclude_labels
                if e in edge_label_map.classes_
            ]
            setattr(self, f"edge_exclude_{label}", edge_label_map.transform(exclude_labels))

            num_labels = len(edge_label_map.classes_) - len(exclude_labels)
            setattr(self, f"num_edges_{label}", num_labels)

            if hasattr(self.graphs[0].data, 'train_edge_mask'):
                validate_classes(self.graphs, label, exclude_labels, 'edge')



    def add_graph_labels(self):
        if hasattr(self.metadata, 'graph'):
            label = self.metadata.graph_cls
            if label and hasattr(self.graphs[0].graph, label):
                graph_labels = list()
                for torch_graph in self.graphs:
                    graph_labels.append(getattr(torch_graph.graph, label))

                graph_label_map = LabelEncoder()
                graph_labels = graph_label_map.fit_transform(graph_labels)

                for torch_graph, graph_label in zip(self.graphs, graph_labels):
                    setattr(torch_graph.data, f"graph_{label}", torch.tensor([graph_label]))
                
                exclude_labels = [
                    graph_label_map.transform([e])[0]
                    for e in self.exclude_labels
                    if e in graph_label_map.classes_
                ]
                setattr(self, f"graph_exclude_{label}", exclude_labels)
                num_labels = len(graph_label_map.classes_) - len(exclude_labels)


                setattr(self, f"num_graph_{label}", num_labels)
                setattr(self, f"graph_label_map_{label}", graph_label_map)
                


    def add_graph_text(self):
        label = self.metadata.graph_label
        for torch_graph in self.graphs:
            if label and hasattr(torch_graph.graph, label):
                setattr(torch_graph, 'text', getattr(torch_graph.graph, label))
            else:
                node_label = self.metadata.node_label
                node_names = [node_data.get(node_label, "") for _, node_data in torch_graph.graph.nodes(data=True)]
                setattr(torch_graph, 'text', doc_tokenizer(" ".join(node_names)))


    def get_torch_geometric_data(self, use_node_types=False, use_edge_types=False):
        def set_types(prefix):
            num_classes = max((max(getattr(g.data, f"{prefix}_type")) for g in self.graphs)) + 1
            print(f"Number of {prefix} types: {num_classes}")
            for g in self.graphs:
                types = torch.nn.functional.one_hot(getattr(g.data, f"{prefix}_type"), num_classes).to(g.data.x.device)
                if prefix == 'node':
                    g.data.x = torch.cat([g.data.x, types], dim=1)
                elif prefix == 'edge':
                    g.data.edge_attr = torch.cat([g.data.edge_attr, types], dim=1)
            
            node_dim = self.graphs[0].data.x.shape[1]
            assert all(g.data.x.shape[1] == node_dim for g in self.graphs), "Node types not added correctly"
            edge_dim = self.graphs[0].data.edge_attr.shape[1]
            assert all(g.data.edge_attr.shape[1] == edge_dim for g in self.graphs), "Edge types not added correctly"

        if use_node_types:
            set_types('node')
        
        if use_edge_types:
            set_types('edge')

        
        return [g.data for g in self.graphs]
    
    
    def get_gnn_graph_classification_data(self):
        train_idx, test_idx = self.get_train_test_split()
        train_data = [self.graphs[i].data for i in train_idx]
        test_data = [self.graphs[i].data for i in test_idx]
        return {
            'train': train_data,
            'test': test_data
        }


    def get_kfold_gnn_graph_classification_data(self):
        for train_idx, test_idx in self.k_fold_split():
            train_data = [self.graphs[i].data for i in train_idx]
            test_data = [self.graphs[i].data for i in test_idx]
            yield {
                'train': train_data,
                'test': test_data,
            }


    def __get_lm_data(self, indices, tokenizer=None, remove_duplicates=False):
        graph_label_name = self.metadata.graph_cls
        assert graph_label_name is not None, "No Graph Label found in data. Please define graph label in metadata"
        X = [getattr(self.graphs[i], 'text') for i in indices]
        y = [getattr(self.graphs[i].data, f'graph_{graph_label_name}')[0].item() for i in indices]
        if tokenizer is None:
            assert self.tokenizer is not None, "Tokenizer is not defined. Please define a tokenizer to tokenize data"
            tokenizer = self.tokenizer

        dataset = EncodingDataset(tokenizer, X, y, remove_duplicates=remove_duplicates)
        return dataset


    def get_lm_graph_classification_data(self, tokenizer=None):
        assert self.metadata.graph_cls, "No Graph Label found in data. Please define graph label in metadata"
        train_idx, test_idx = self.get_train_test_split()
        train_dataset = self.__get_lm_data(train_idx, tokenizer)
        test_dataset = self.__get_lm_data(test_idx, tokenizer)

        return {
            'train': train_dataset,
            'test': test_dataset,
            'num_classes': getattr(self, f'num_graph_{self.metadata.graph_cls}')
        }
        
    
    def get_kfold_lm_graph_classification_data(self, tokenizer=None, remove_duplicates=True):
        assert self.metadata.graph_cls, "No Graph Label found in data. Please define graph label in metadata"
        for train_idx, test_idx in self.k_fold_split():
            train_dataset = self.__get_lm_data(train_idx, tokenizer, remove_duplicates)
            test_dataset = self.__get_lm_data(test_idx, tokenizer, remove_duplicates)
            yield {
                'train': train_dataset,
                'test': test_dataset,
                'num_classes': getattr(self, f'num_graph_{self.metadata.graph_cls}')
            }


class GraphEdgeDataset(GraphDataset):
    def __init__(
            self, 
            models_dataset: Union[EcoreModelDataset, ArchiMateModelDataset],
            save_dir='datasets/graph_data',
            distance=1,
            reload=False,
            test_ratio=0.2,
            add_negative_train_samples=False,
            neg_sampling_ratio=1,
            use_embeddings=False,
            use_special_tokens=False,
            use_attributes=False,
            use_edge_types=False,
            use_node_types=False,
            embed_model_name='bert-base-uncased',
            ckpt=None,
            no_shuffle=False,
            randomize_ne = False,
            randomize_ee = False,
            random_embed_dim=768,
            tokenizer_special_tokens=None
        ):
        super().__init__(
            models_dataset=models_dataset,
            save_dir=save_dir,
            distance=distance,
            test_ratio=test_ratio,
            use_embeddings=use_embeddings,
            use_special_tokens=use_special_tokens,
            embed_model_name=embed_model_name,
            ckpt=ckpt,
            use_edge_types=use_edge_types,
            use_node_types=use_node_types,
            use_attributes=use_attributes,
            no_shuffle=no_shuffle,
            randomize_ne=randomize_ne,
            randomize_ee=randomize_ee,
            random_embed_dim=random_embed_dim,
            tokenizer_special_tokens=tokenizer_special_tokens
        )

        save_path = os.path.join(self.save_dir, 'edge_data')
        save_path += f'/{models_dataset.name}'

        save_path += f'_dist_{distance}'
        save_path += '_neg' if add_negative_train_samples else ''
        save_path += f'_nsr_{int(neg_sampling_ratio)}' if add_negative_train_samples else ''
        save_path += f'_et_{int(use_edge_types)}' if use_edge_types else ''
        save_path += f'_attr_{int(use_attributes)}' if use_attributes else ''
        save_path += f'_nt_{int(use_node_types)}' if use_node_types else ''
        save_path += f'_sp_{int(use_special_tokens)}' if use_special_tokens else ''
        save_path += f'_use_emb_{int(use_embeddings)}' if use_embeddings else ''
        save_path += f'_ckpt_{ckpt.replace("/", "_")}' if ckpt else ''
        save_path += f'_test_{test_ratio}'
        save_path += '.pkl'

        if use_attributes:
            assert self.metadata.node_attributes is not None, "Node attributes are not defined in metadata to be used"

        if os.path.exists(save_path) and not reload:
            self.load(save_path)
        else:
            self.graphs = [
                TorchEdgeGraph(
                    g, 
                    save_dir=self.save_dir,
                    metadata=self.metadata,
                    distance=distance,
                    test_ratio=test_ratio,

                    embedder=self.embedder,

                    use_neg_samples=add_negative_train_samples,
                    neg_samples_ratio=neg_sampling_ratio,
                    use_edge_types=use_edge_types,
                    use_node_types=use_node_types,
                    use_attributes=use_attributes,
                    use_special_tokens=use_special_tokens,
                ) 
                for g in tqdm(models_dataset, desc='Creating graphs')
            ]

            self.save(save_path)

        self.post_process_graphs()
        train_count, test_count = dict(), dict()
        for g in self.graphs:
            train_mask = g.data.train_edge_mask
            test_mask = g.data.test_edge_mask
            train_labels = getattr(g.data, f'edge_{self.metadata.edge_cls}')[train_mask]
            test_labels = getattr(g.data, f'edge_{self.metadata.edge_cls}')[test_mask]
            t1 = dict(Counter(train_labels.tolist()))
            t2 = dict(Counter(test_labels.tolist()))
            for k in t1:
                train_count[k] = train_count.get(k, 0) + t1[k]
            
            for k in t2:
                test_count[k] = test_count.get(k, 0) + t2[k]

        print(f"Train edge classes: {train_count}")
        print(f"Test edge classes: {test_count}")

    def get_link_prediction_texts(
        self, 
        distance=1, 
        label: str = 'type',
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

            # print(train_neg_edge_index.shape)

            edge_indices = {
                'train_pos': train_pos_edge_index,
                'train_neg': train_neg_edge_index,
                'test_pos': test_pos_edge_index,
                'test_neg': test_neg_edge_index
            }

            for edge_index_label, edge_index in edge_indices.items():
                edge_strs = graph.get_graph_edge_strs_from_node_strs(
                    node_strs, 
                    edge_index,
                    use_edge_types=self.use_edge_types,
                    neg_samples="neg" in edge_index_label,
                    use_special_tokens=self.use_special_tokens
                )
                
                edge_strs = list(edge_strs.values())
                data[f'{edge_index_label}_edges'] += edge_strs


            if task_type == LP_TASK_EDGE_CLS:
                train_mask = graph.data.train_edge_mask
                test_mask = graph.data.test_edge_mask
                data['train_edge_classes'] += getattr(graph.data, f'edge_{label}')[train_mask]
                data['test_edge_classes'] += getattr(graph.data, f'edge_{label}')[test_mask]


        print(data[f'train_pos_edges'][:20])
        print(data[f'test_pos_edges'][:20])

        edge_label_map = getattr(self, f"edge_label_map_{label}")

        print(edge_label_map.inverse_transform([i.item() for i in data[f'train_edge_classes'][:20]]))
        print(edge_label_map.inverse_transform([i.item() for i in data[f'test_edge_classes'][:20]]))
        return data
    

    def get_link_prediction_lm_data(
        self, 
        tokenizer: AutoTokenizer,
        distance, 
        label: str = None,
        task_type=LP_TASK_EDGE_CLS
    ):
        data = self.get_link_prediction_texts(
            distance, 
            label, 
            task_type
        )


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
            use_special_tokens=False,

            use_embeddings=False,
            embed_model_name='bert-base-uncased',
            ckpt=None,

            no_shuffle=False,
            randomize_ne = False,
            randomize_ee = False,

            random_embed_dim=768,

            tokenizer_special_tokens=None
        ):
        super().__init__(
            models_dataset=models_dataset,
            save_dir=save_dir,
            distance=distance,
            test_ratio=test_ratio,
            use_embeddings=use_embeddings,
            use_special_tokens=use_special_tokens,
            use_edge_types=use_edge_types,
            embed_model_name=embed_model_name,
            ckpt=ckpt,
            no_shuffle=no_shuffle,
            randomize_ne=randomize_ne,
            randomize_ee=randomize_ee,
            random_embed_dim=random_embed_dim,
            tokenizer_special_tokens=tokenizer_special_tokens
        )

        save_path = os.path.join(self.save_dir, 'node_data')
        save_path += f'_dist_{distance}'
        save_path += f'_et_{int(use_edge_types)}' if use_edge_types else ''
        save_path += f'_attr_{int(use_attributes)}' if use_attributes else ''
        save_path += f'_sp_{int(use_special_tokens)}' if use_special_tokens else ''
        save_path += f'_use_emb_{int(use_embeddings)}' if use_embeddings else ''
        save_path += f'_ckpt_{ckpt}' if ckpt else ''
        save_path += f'_test_{test_ratio}'
        save_path += '.pkl'

        if use_attributes:
            assert self.metadata.node_attributes is not None, "Node attributes are not defined in metadata to be used"

        if os.path.exists(save_path) and not reload:
            self.load(save_path)
        else:
            self.graphs = [
                TorchNodeGraph(
                    g, 
                    metadata=self.metadata,
                    save_dir=self.save_dir,
                    distance=distance,
                    test_ratio=test_ratio,
                    embedder=self.embedder,
                    use_attributes=use_attributes,
                    use_edge_types=use_edge_types,
                    use_special_tokens=use_special_tokens,
                ) 
                for g in tqdm(models_dataset, desc='Creating graphs')
            ]

            self.save(save_path)
        
        self.post_process_graphs()

        node_labels = self.metadata.node_cls
        if isinstance(node_labels, str):
            node_labels = [node_labels]

        for node_label in node_labels:
            print(f"Node label: {node_label}")
            train_count, test_count = dict(), dict()
            for g in self.graphs:
                train_idx = g.data.train_node_mask
                test_idx = g.data.test_node_mask
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


    def get_node_classification_texts(self, distance, label):
        data = {'train_nodes': [], 'train_node_classes': [], 'test_nodes': [], 'test_node_classes': []}
        for graph in tqdm(self.graphs, desc='Getting node classification data'):
            node_strs = list(graph.get_graph_node_strs(graph.data.edge_index, distance).values())

            train_node_strs = [node_strs[i.item()] for i in torch.where(graph.data.train_node_mask)[0]]
            test_node_strs = [node_strs[i.item()] for i in torch.where(graph.data.test_node_mask)[0]]
            
            train_node_classes = getattr(graph.data, f'node_{label}')[graph.data.train_node_mask]
            test_node_classes = getattr(graph.data, f'node_{label}')[graph.data.test_node_mask]

            exclude_labels = getattr(self, f'node_exclude_{label}')
            train_node_strs, train_node_classes = exclude_labels_from_data(train_node_strs, train_node_classes, exclude_labels)
            test_node_strs, test_node_classes = exclude_labels_from_data(test_node_strs, test_node_classes, exclude_labels)
            
            data['train_nodes'] += train_node_strs
            data['train_node_classes'] += train_node_classes
            data['test_nodes'] += test_node_strs
            data['test_node_classes'] += test_node_classes
        

        print("Tokenizing data")
        print(data['train_nodes'][:10])
        print(data['train_node_classes'][:10])
        print(data['test_nodes'][:10])
        print(data['test_node_classes'][:10])
        print(len(data['train_nodes']))
        print(len(data['train_node_classes']))
        print(len(data['test_nodes']))
        print(len(data['test_node_classes']))
        # exit(0)
        return data


    def get_node_classification_lm_data(
        self, 
        label: str,
        tokenizer: AutoTokenizer,
        distance: int = 1, 
    ):
        data = self.get_node_classification_texts(distance, label)
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
    

def get_models_gpt_dataset(
        models_dataset: Union[ArchiMateModelDataset, EcoreModelDataset], 
        tokenizer: AutoTokenizer,
        chunk_size: int = 100,
        chunk_overlap: int = 20,
        max_length: int = 128,
        **config_params
    ):

    def split_texts_into_chunks(
        texts: List[str],
        size: int = 100,
        overlap: int = 20,
    ):
        text_splitter = RecursiveCharacterTextSplitter(
            # Set a really small chunk size, just to show.
            chunk_size=size,
            chunk_overlap=overlap,
            length_function=len,
            is_separator_regex=False,
        )
        return [t.page_content for t in text_splitter.create_documents(texts)]
    
    graph_dataset = GraphEdgeDataset(models_dataset, **config_params)
    texts_data = graph_dataset.get_link_prediction_texts()
    texts = texts_data['train_pos_edges'] + texts_data['test_pos_edges']

    print("Total texts", len(texts))
    splitted_texts = split_texts_into_chunks(
        texts, 
        size=chunk_size, 
        overlap=chunk_overlap
    )
    print(len(splitted_texts))
    print("Tokenizing...")
    dataset = GPTTextDataset(texts, tokenizer, max_length=max_length)
    print("Tokenized")
    print("Max length", dataset[:]['input_ids'].shape)
    return dataset