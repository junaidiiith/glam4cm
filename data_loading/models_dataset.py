from typing import List
import pandas as pd
from tqdm.auto import tqdm
import pickle
from random import shuffle
from sklearn.model_selection import StratifiedKFold
import json
import os
from data_loading.encoding import EncodingDataset
from lang2graph.archimate import ArchiMateNxG
from lang2graph.ecore import EcoreNxG
from lang2graph.common import LangGraph
from settings import (
    datasets_dir, 
    seed,
)
import numpy as np


from settings import logger


class ModelDataset:
    def __init__(
        self,
        dataset_name: str,
        dataset_dir=datasets_dir,
        save_dir='datasets/pickles',
        min_edges: int = -1,
        min_enr: float = -1,
        timeout=-1,
        preprocess_graph_text: callable = None
    ):
        self.name = dataset_name
        self.dataset_dir = dataset_dir
        self.save_dir = save_dir
        os.makedirs(save_dir, exist_ok=True)

        self.min_edges = min_edges
        self.min_enr = min_enr
        self.timeout = timeout
        self.preprocess_graph_text = preprocess_graph_text


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
        
        if self.preprocess_graph_text:
            X = [self.preprocess_graph_text(x) for x in X]
        return X, y
    
    def __get_lm_data(self, train_idx, test_idx, tokenizer, remove_duplicates=False):
        X, y = self.data
        y_enc = {label: i for i, label in enumerate(set(y))}
        y = [y_enc[label] for label in y]
        X_train, y_train = [X[i] for i in train_idx], [y[i] for i in train_idx]
        X_test, y_test = [X[i] for i in test_idx], [y[i] for i in test_idx]
        train_dataset = EncodingDataset(tokenizer, X_train, y_train, remove_duplicates=remove_duplicates)
        test_dataset = EncodingDataset(tokenizer, X_test, y_test, remove_duplicates=remove_duplicates)
        num_classes = len(set(y))
        return {
            'train': train_dataset,
            'test': test_dataset,
            'num_classes': num_classes
        }

    def get_graph_classification_data(self, tokenizer, remove_duplicates=False):
        train_idx, test_idx = self.get_train_test_split()
        return self.__get_lm_data(train_idx, test_idx, tokenizer, remove_duplicates=remove_duplicates)
    
    def get_graph_classification_data_kfold(self, tokenizer, k=10, remove_duplicates=False):
        for train_idx, test_idx in self.k_fold_split(k=k):
            yield self.__get_lm_data(train_idx, test_idx, tokenizer, remove_duplicates=remove_duplicates)


    def __repr__(self):
        return f'Dataset({self.name}, graphs={len(self.graphs)})'
    
    def __getitem__(self, key) -> LangGraph:
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


    def filter_graphs(self):
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
        
        self.filter_graphs()
        print(f'Loaded {self.name} with {len(self.graphs)} graphs')


class EcoreModelDataset(ModelDataset):
    def __init__(
            self, 
            dataset_name: str, 
            dataset_dir=datasets_dir,
            save_dir='datasets/pickles',
            reload=False,
            remove_duplicates=False,
            min_edges: int = -1,
            min_enr: float = -1,
            preprocess_graph_text: callable = None
        ):
        super().__init__(
            dataset_name, 
            dataset_dir=dataset_dir, 
            save_dir=save_dir, 
            min_edges=min_edges, 
            min_enr=min_enr,
            preprocess_graph_text=preprocess_graph_text
        )
        os.makedirs(save_dir, exist_ok=True)

        dataset_exists = os.path.exists(os.path.join(save_dir, f'{dataset_name}.pkl'))
        if reload or not dataset_exists:
            self.graphs: List[EcoreNxG] = []
            data_path = os.path.join(dataset_dir, dataset_name)
            for file in os.listdir(data_path):
                if file.endswith('.jsonl') and file.startswith('ecore'):
                    json_objects = json.load(open(os.path.join(data_path, file)))
                    for g in tqdm(json_objects, desc=f'Loading {dataset_name.title()}'):
                        if remove_duplicates and g['is_duplicated']:
                            continue
                        nxg = EcoreNxG(g)
                        self.graphs.append(nxg)

            print(f'Loaded Total {self.name} with {len(self.graphs)} graphs')
            print("Filtering...")
            self.save()
            self.filter_graphs()
        else:
            self.load()
        
        logger.info(f'Loaded {self.name} with {len(self.graphs)} graphs')
        
        if remove_duplicates:
            self.dedup()

        logger.info(f'Graphs: {len(self.graphs)}')
        print(f'Loaded {self.name} with {len(self.graphs)} graphs')


    def dedup(self) -> List[EcoreNxG]:
        logger.info(f'Deduplicating {self.name}')
        return [g for g in self.graphs if not g.is_duplicated]



class ArchiMateModelDataset(ModelDataset):
    def __init__(
            self, 
            dataset_name: str, 
            dataset_dir=datasets_dir,
            save_dir='datasets/pickles',
            reload=False,
            remove_duplicates=False,
            min_edges: int = -1,
            min_enr: float = -1,
            timeout=-1,
            language=None,
            preprocess_graph_text: callable = None
        ):
        super().__init__(
            dataset_name, 
            dataset_dir=dataset_dir, 
            save_dir=save_dir, 
            min_edges=min_edges, 
            min_enr=min_enr,
            timeout=timeout,
            preprocess_graph_text=preprocess_graph_text
        )
        os.makedirs(save_dir, exist_ok=True)
        
        dataset_exists = os.path.exists(os.path.join(save_dir, f'{dataset_name}.pkl'))
        if reload or not dataset_exists:
            self.graphs: List[ArchiMateNxG] = []
            data_path = os.path.join(dataset_dir, dataset_name, 'processed-models')
            if language:
                df = pd.read_csv(os.path.join(dataset_dir, dataset_name, f'{language}-metadata.csv'))
                model_dirs = df['ID'].to_list()
            else:
                model_dirs = os.listdir(data_path)

            for model_dir in tqdm(model_dirs, desc=f'Loading {dataset_name.title()}'):
                model_dir = os.path.join(data_path, model_dir)
                if os.path.isdir(model_dir):
                    model_file = os.path.join(model_dir, 'model.json')
                    if os.path.exists(model_file):
                        model = json.load(open(model_file))
                        try:
                            nxg = ArchiMateNxG(
                                model, 
                                path=model_file,
                                timeout=timeout
                            )
                            if nxg.number_of_edges() < 1:
                                continue
                            self.graphs.append(nxg)
                            
                        except Exception as e:
                            raise e
                
            self.filter_graphs()
            self.save()
        else:
            self.load()
        
        if remove_duplicates:
            self.dedup()
        
        print(f'Loaded {self.name} with {len(self.graphs)} graphs')
        print(f'Graphs: {len(self.graphs)}')
    

    def dedup(self) -> List[ArchiMateNxG]:
        return list({str(g.edges(data=True)): g for g in self.graphs}.values())