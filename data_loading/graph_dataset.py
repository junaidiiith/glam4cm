import json
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
from data_loading.data import TorchEdgeGraph, TorchGraph, TorchNodeGraph, GraphData
from data_loading.models_dataset import ArchiMateModelDataset, EcoreModelDataset
from data_loading.encoding import EncodingDataset, GPTTextDataset
from tqdm.auto import tqdm
from embeddings.w2v import Word2VecEmbedder
from embeddings.tfidf import TfidfEmbedder
from embeddings.common import get_embedding_model
from lang2graph.common import LangGraph, get_node_data, get_edge_data
from data_loading.metadata import ArchimateMetaData, EcoreMetaData
from settings import seed
from settings import (
    LP_TASK_EDGE_CLS,
    LP_TASK_LINK_PRED,
)
from tokenization.utils import doc_tokenizer, get_tokenizer
import utils


def exclude_labels_from_data(X, labels, exclude_labels):
    X = [X[i] for i in range(len(X)) if labels[i] not in exclude_labels]
    labels = [labels[i] for i in range(len(labels)) if labels[i] not in exclude_labels]
    return X, labels


def validate_classes(torch_graphs: List[TorchGraph], label, exclude_labels, element):
    train_labels, test_labels = list(), list()
    train_idx_label = f"train_{element}_mask"
    test_idx_label = f"test_{element}_mask"

    for torch_graph in tqdm(torch_graphs, desc=f"Validating {element} classes"):
        labels = getattr(torch_graph.data, f"{element}_{label}")
        train_idx = getattr(torch_graph.data, train_idx_label)
        test_idx = getattr(torch_graph.data, test_idx_label)
        indices = np.nonzero(np.isin(labels, exclude_labels))[0]
        
        if len(indices) > 0:
            train_idx = train_idx[~np.isin(train_idx, indices)]
            test_idx = test_idx[~np.isin(test_idx, indices)]

        train_labels.append(labels[train_idx])
        test_labels.append(labels[test_idx])


        edge_classes = [t for _, _, t in torch_graph.graph.edges(data=label)]
        node_classes = [t for _, t in torch_graph.graph.nodes(data=label)]
        
        if element == 'edge':
            for idx in train_idx.tolist() + test_idx.tolist():
                t_c = edge_classes[idx]
                edge = torch_graph.graph.idx_to_edge[idx]
                assert torch_graph.graph.numbered_graph.edges[edge][label] == t_c, f"Edge {label} mismatch for {edge}"

        elif element == 'node':
            for idx in train_idx.tolist() + test_idx.tolist():
                t_c = node_classes[idx]
                node_label = torch_graph.graph.id_to_node_label[idx]
                try:
                    assert torch_graph.graph.nodes[node_label][label] == t_c, f"Node {label} mismatch for {idx}"
                except AssertionError as e:
                    raise e
        else:
            raise ValueError(f"Invalid element: {element}")


    train_classes = set(sum([
        getattr(torch_graph.data, f"{element}_{label}")[getattr(torch_graph.data, train_idx_label)].tolist() 
        for torch_graph in torch_graphs], []
    ))
    test_classes = set(sum([
        getattr(torch_graph.data, f"{element}_{label}")[getattr(torch_graph.data, test_idx_label)].tolist() 
        for torch_graph in torch_graphs], []
    ))
    num_train_classes = len(train_classes)
    num_test_classes = len(test_classes)
    print("Train classes:", train_classes)
    print("Test classes:", test_classes)
    print(f"Number of classes in training set: {num_train_classes}")
    print(f"Number of classes in test set: {num_test_classes}")
    
    

class GraphDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        models_dataset: Union[EcoreModelDataset, ArchiMateModelDataset],
        save_dir='datasets/graph_data',
        distance=1,
        add_negative_train_samples=False,
        neg_sampling_ratio=1,
        use_attributes=False,
        use_edge_types=False,
        use_node_types=False,
        use_edge_label=False,
        no_labels=False,

        node_cls_label=None,
        edge_cls_label='type',
        
        test_ratio=0.2,

        use_embeddings=False,
        use_special_tokens=False,
        embed_model_name='bert-base-uncased',
        ckpt=None,
        reload=False,
        no_shuffle=False,
        
        randomize_ne=False,
        randomize_ee=False,
        random_embed_dim=128,
        
        exclude_labels: list = [None]
    ):
        if isinstance(models_dataset, EcoreModelDataset):
            self.metadata = EcoreMetaData()
        elif isinstance(models_dataset, ArchiMateModelDataset):
            self.metadata = ArchimateMetaData()

        self.distance = distance
        self.use_embeddings = use_embeddings
        self.ckpt = ckpt
        self.embedder = get_embedding_model(embed_model_name, ckpt) if use_embeddings else None

        self.reload = reload

        self.use_edge_types = use_edge_types
        self.use_node_types = use_node_types
        self.use_attributes = use_attributes
        self.use_edge_label = use_edge_label
        self.no_labels = no_labels
        
        self.add_negative_train_samples = add_negative_train_samples
        self.neg_sampling_ratio = neg_sampling_ratio

        self.test_ratio = test_ratio

        self.no_shuffle = no_shuffle
        self.exclude_labels = exclude_labels

        self.use_special_tokens = use_special_tokens
        self.node_cls_label = node_cls_label
        self.edge_cls_label = edge_cls_label

        self.randomize_ne = randomize_ne
        self.randomize_ee = randomize_ee
        self.random_embed_dim = random_embed_dim

        self.graphs: List[Union[TorchNodeGraph, TorchEdgeGraph]] = []
        self.config = dict(
            name=models_dataset.name,
            distance=distance,
            add_negative_train_samples=add_negative_train_samples,
            neg_sampling_ratio=neg_sampling_ratio,
            use_attributes=use_attributes,
            use_edge_types=use_edge_types,
            use_node_types=use_node_types,
            use_edge_label=use_edge_label,
            no_labels=no_labels,
            use_special_tokens=use_special_tokens,
            use_embeddings=use_embeddings,
            embed_model_name=embed_model_name if use_embeddings else None,
            ckpt=ckpt if use_embeddings else None,
            no_shuffle=no_shuffle,
            exclude_labels=exclude_labels,
            node_cls_label=node_cls_label,
            edge_cls_label=edge_cls_label,
            test_ratio=test_ratio,
            randomize_ne=randomize_ne,
            randomize_ee=randomize_ee,
            random_embed_dim=random_embed_dim
        )

        self.save_dir = os.path.join(save_dir, models_dataset.name)
        os.makedirs(self.save_dir, exist_ok=True)

        self.file_paths = {
            graph.hash: os.path.join(self.save_dir, f'{graph.hash}', 'data.pkl') 
            for graph in models_dataset
        }


    def get_config_hash(self):
        if os.path.exists(os.path.join(self.save_dir, 'configs.json')):
            with open(os.path.join(self.save_dir, 'configs.json'), 'r') as f:
                configs = json.load(f)
        else:
            configs = dict()
        
        config_hash = utils.md5_hash(str(self.config))
        if config_hash not in configs:
            configs[config_hash] = self.config
            with open(os.path.join(self.save_dir, 'configs.json'), 'w') as f:
                json.dump(configs, f)
        
        return config_hash
    

    def set_file_hashes(self, models_dataset: Union[EcoreModelDataset, ArchiMateModelDataset]):
        self.config_hash = self.get_config_hash()
        os.makedirs(os.path.join(self.save_dir, self.config_hash), exist_ok=True)
        self.file_paths = {
            graph.hash: os.path.join(self.save_dir, self.config_hash, f'{graph.hash}', 'data.pkl') 
            for graph in models_dataset
        }

    def get_torch_graphs(
            self, 
            type: str,
            models_dataset: Union[EcoreModelDataset, ArchiMateModelDataset], 
            limit: int =-1
        ) -> List[tuple[TorchGraph, str]]:
        
        common_params = dict(
            metadata=self.metadata,
            distance=self.distance,
            test_ratio=self.test_ratio,
            use_attributes=self.use_attributes,
            use_node_types=self.use_node_types,
            use_edge_types=self.use_edge_types,
            use_edge_label=self.use_edge_label,
            use_special_tokens=self.use_special_tokens,
            no_labels=self.no_labels,
            node_cls_label=self.node_cls_label,
            edge_cls_label=self.edge_cls_label
        )
        def create_node_graph(graph: LangGraph):
            node_params = {
                **common_params,
            }
            torch_graph = TorchNodeGraph(graph, **node_params)
            return torch_graph
        
        def create_edge_graph(graph: LangGraph):
            edge_params = {
                **common_params,
                'add_negative_train_samples': self.add_negative_train_samples,
                'neg_samples_ratio': self.neg_sampling_ratio
            }
            torch_graph = TorchEdgeGraph(graph, **edge_params)
            return torch_graph
        
        self.set_file_hashes(models_dataset)
        
        models_size = len(models_dataset) \
            if (limit == -1 or limit > len(models_dataset)) else limit
        
        models_dataset = models_dataset[:models_size]
        torch_graphs = list()
        for graph in tqdm(models_dataset, desc=f'Creating {type} graphs'):
            fp = self.file_paths[graph.hash]
            if type == 'node':
                torch_graph = create_node_graph(graph)
            elif type == 'edge':
                torch_graph = create_edge_graph(graph)
            torch_graphs.append((torch_graph, fp))
        
        return torch_graphs

    
    def embed(self, torch_graphs: List[tuple[TorchGraph, str]]):
        for torch_graph, fp in tqdm(torch_graphs, desc='Embedding graphs'):
            torch_graph.embed(
                self.embedder, 
                save_path=os.path.join(fp),
                reload=self.reload,
                randomize_ne=self.randomize_ne,
                randomize_ee=self.randomize_ee,
                random_embed_dim=self.random_embed_dim
            )

        for torch_graph, fp in tqdm(torch_graphs, desc='Re-Loading graphs'):
            tg = TorchGraph.load(fp)
            self.graphs.append(tg)
        
        if not self.no_shuffle:
            shuffle(self.graphs)

        self.post_process_graphs()
        self.validate_graphs()


    def post_process_graphs(self):
        self.add_cls_labels()

        def set_types(prefix):
            prefix_cls = getattr(self, f"{prefix}_cls_label")
            num_classes = getattr(self, f"num_{prefix}s_{prefix_cls}") + 1
            # print(f"Number of {prefix} types: {num_classes}")
            for g in self.graphs:
                types = np.eye(num_classes)[getattr(g.data, f"{prefix}_{prefix_cls}")]
                
                if prefix == 'node':
                    g.data.x = np.concatenate([g.data.x, types], axis=1)
                elif prefix == 'edge':
                    g.data.edge_attr = np.concatenate([g.data.edge_attr, types], axis=1)
                    
            node_dim = self.graphs[0].data.x.shape[1]
            assert all(g.data.x.shape[1] == node_dim for g in self.graphs), "Node types not added correctly"
            edge_dim = self.graphs[0].data.edge_attr.shape[1]
            assert all(g.data.edge_attr.shape[1] == edge_dim for g in self.graphs), "Edge types not added correctly"


        if self.use_node_types and self.node_cls_label:
            set_types('node')
        
        if self.use_edge_types and self.edge_cls_label:
            set_types('edge')


    def __len__(self):
        return len(self.graphs)
    

    def __getitem__(self, index: int):
        return self.graphs[index]


    def get_torch_dataset(self):
        return [g.data.to_graph_data() for g in self.graphs]


    def save(self, save_path):
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        print(f"Saving data to {save_path}")
        with open(save_path, 'wb') as f:
            pickle.dump(self.graphs, f)


    def load(self, save_path):
        print(f"Loading data from {save_path}")
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
    

    def validate_graphs(self):
        for torch_graph in self.graphs:
            assert torch_graph.data.x.shape[0] == torch_graph.graph.number_of_nodes(), \
            f"Number of nodes mismatch, {torch_graph.data.x.shape[0]} != {torch_graph.graph.number_of_nodes()}"
            
            if isinstance(torch_graph, TorchEdgeGraph):
                assert torch_graph.data.overall_edge_index.shape[1] == torch_graph.graph.number_of_edges(), \
                f"Number of edges mismatch, {torch_graph.data.edge_index.shape[1]} != {torch_graph.graph.number_of_edges()}"
            else:
                assert torch_graph.data.edge_index.shape[1] == torch_graph.graph.number_of_edges(), \
                f"Number of edges mismatch, {torch_graph.data.edge_index.shape[1]} != {torch_graph.graph.number_of_edges()}"

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
                setattr(torch_graph.data, f"node_{label}", np.array(node_classes))
            
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
                setattr(torch_graph.data, f"edge_{label}", np.array(edge_classes))
            
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
                    setattr(torch_graph.data, f"graph_{label}", np.array([graph_label]))
                
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


    def __get_lm_data(self, indices, tokenizer, remove_duplicates=False):
        graph_label_name = self.metadata.graph_cls
        assert graph_label_name is not None, "No Graph Label found in data. Please define graph label in metadata"
        X = [getattr(self.graphs[i], 'text') for i in indices]
        y = [getattr(self.graphs[i].data, f'graph_{graph_label_name}')[0].item() for i in indices]

        dataset = EncodingDataset(tokenizer, X, y, remove_duplicates=remove_duplicates)
        return dataset


    def get_lm_graph_classification_data(self, tokenizer):
        assert self.metadata.graph_cls, "No Graph Label found in data. Please define graph label in metadata"
        train_idx, test_idx = self.get_train_test_split()
        train_dataset = self.__get_lm_data(train_idx, tokenizer)
        test_dataset = self.__get_lm_data(test_idx, tokenizer)

        return {
            'train': train_dataset,
            'test': test_dataset,
            'num_classes': getattr(self, f'num_graph_{self.metadata.graph_cls}')
        }
        
    
    def get_kfold_lm_graph_classification_data(self, tokenizer, remove_duplicates=True):
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
            distance=0,
            reload=False,
            test_ratio=0.2,

            add_negative_train_samples=False,
            neg_sampling_ratio=1,

            use_attributes=False,
            use_edge_types=False,
            use_edge_label=False,
            use_node_types=False,
            no_labels=False,
            
            use_embeddings=False,
            embed_model_name='bert-base-uncased',
            ckpt=None,

            no_shuffle=False,
            randomize_ne = False,
            randomize_ee = False,
            random_embed_dim=128,

            use_special_tokens=False,

            limit: int = -1,

            node_cls_label: str = None,
            edge_cls_label: str = 'type',

            task_type=LP_TASK_EDGE_CLS
        ):
        
        super().__init__(
            models_dataset=models_dataset,
            save_dir=save_dir,
            distance=distance,
            test_ratio=test_ratio,

            use_node_types=use_node_types,
            use_edge_types=use_edge_types,
            use_edge_label=use_edge_label,
            use_attributes=use_attributes,
            no_labels=no_labels,

            add_negative_train_samples=add_negative_train_samples,
            neg_sampling_ratio=neg_sampling_ratio,
            
            use_special_tokens=use_special_tokens,
            

            node_cls_label=node_cls_label,
            edge_cls_label=edge_cls_label,

            use_embeddings=use_embeddings,
            embed_model_name=embed_model_name,
            ckpt=ckpt,

            reload=reload,
            no_shuffle=no_shuffle,
            
            
            randomize_ne=randomize_ne,
            randomize_ee=randomize_ee,
            random_embed_dim=random_embed_dim,
        )

        self.task_type = task_type

        torch_graphs = self.get_torch_graphs('edge', models_dataset, limit)


        if self.use_embeddings and (isinstance(self.embedder, Word2VecEmbedder) or isinstance(self.embedder, TfidfEmbedder)):
            texts = self.get_link_prediction_texts()
            texts = sum([v for _, v in texts.items() if not v.endswith("classes")], [])
            print(f"Training {self.embedder.name} Embedder")
            self.embedder.train(texts)
            print(f"Trained {self.embedder.name} Embedder")
        
        self.embed(torch_graphs)

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
        label: str = None
    ):
        if label is None:
            label = self.edge_cls_label

        data = defaultdict(list)
        for graph in tqdm(self.graphs, desc=f'Getting {self.task_type} data'):
            graph_data = graph.get_link_prediction_texts(label, self.task_type)
            for k, v in graph_data.items():
                data[k] += v

        print("Train Texts: ", data[f'train_pos_edges'][:20])
        print("Test Texts: ", data[f'test_pos_edges'][:20])

        # print("Train Classes", edge_label_map.inverse_transform([i.item() for i in data[f'train_edge_classes'][:20]]))
        # print("Test Classes", edge_label_map.inverse_transform([i.item() for i in data[f'test_edge_classes'][:20]]))
        return data
    

    def get_link_prediction_lm_data(
        self, 
        tokenizer: AutoTokenizer,
        label: str = None,
        task_type=None
    ):
        if label is None:
            label = self.edge_cls_label
        
        if task_type is None:
            task_type = self.task_type

        data = self.get_link_prediction_texts(
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
        distance=0,
        test_ratio=0.2,
        reload=False,
        
        use_attributes=False,
        use_edge_types=False,
        use_node_types=False,
        use_edge_label=False,
        use_special_tokens=False,

        use_embeddings=False,
        embed_model_name='bert-base-uncased',
        ckpt=None,

        no_shuffle=False,
        randomize_ne=False,
        randomize_ee=False,
        random_embed_dim=128,

        limit: int = -1,
        no_labels=False,
        node_cls_label: str = None,
        edge_cls_label: str = 'type'
    ):
        super().__init__(
            models_dataset=models_dataset,
            save_dir=save_dir,
            distance=distance,
            test_ratio=test_ratio,
            
            
            
            use_node_types=use_node_types,
            use_edge_types=use_edge_types,
            use_edge_label=use_edge_label,
            use_attributes=use_attributes,
            no_labels=no_labels,

            node_cls_label=node_cls_label,
            edge_cls_label=edge_cls_label,

            use_embeddings=use_embeddings,
            embed_model_name=embed_model_name,
            ckpt=ckpt,

            reload=reload,
            no_shuffle=no_shuffle,
            
            use_special_tokens=use_special_tokens,

            randomize_ne=randomize_ne,
            randomize_ee=randomize_ee,
            random_embed_dim=random_embed_dim,
        )

        torch_graphs = self.get_torch_graphs('node', models_dataset, limit)

        if self.use_embeddings and (isinstance(self.embedder, Word2VecEmbedder) or isinstance(self.embedder, TfidfEmbedder)):
            texts = self.get_node_classification_texts()
            texts = sum([v for _, v in texts.items() if not v.endswith("classes")], [])
            print(f"Training {self.embedder.name} Embedder")
            self.embedder.train(texts)
            print(f"Trained {self.embedder.name} Embedder")
        
        self.embed(torch_graphs)


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


    def get_node_classification_texts(self, distance=None, label=None):
        if distance is None:
            distance = self.distance

        label = self.metadata.node_cls if label is None else label

        if isinstance(label, list):
            label = label[0]

        node_label_map = getattr(self, f"node_label_map_{label}")

        data = {'train_nodes': [], 'train_node_classes': [], 'test_nodes': [], 'test_node_classes': []}
        for graph in tqdm(self.graphs, desc='Getting node classification data'):
            node_strs = list(graph.get_graph_node_strs(graph.data.edge_index, distance).values())

            train_node_strs = [node_strs[i.item()] for i in graph.data.train_node_mask]
            test_node_strs = [node_strs[i.item()] for i in graph.data.test_node_mask]
            
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
        print(data['test_nodes'][:10])
        if hasattr(self, "node_label_map_type"):
            node_label_map.inverse_transform([i.item() for i in train_node_classes]) == train_node_strs
            node_label_map.inverse_transform([i.item() for i in test_node_classes]) == test_node_strs
            
        print(len(data['train_nodes']))
        print(len(data['train_node_classes']))
        print(len(data['test_nodes']))
        print(len(data['test_node_classes']))
        return data


    def get_node_classification_lm_data(
        self, 
        label: str,
        tokenizer: AutoTokenizer,
        distance: int = 0, 
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