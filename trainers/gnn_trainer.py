from abc import abstractmethod
import torch
from typing import Union
import pandas as pd

from models.gnn_layers import (
    GNNConv, 
    EdgeClassifer,
    NodeClassifier
)
from utils import get_device
from itertools import chain
from tqdm.auto import tqdm
import torch.nn as nn
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.optim import Adam

from sklearn.metrics import (
    balanced_accuracy_score,
    f1_score, 
    roc_auc_score,
    accuracy_score
)


device = get_device()


class Trainer:
    """
    Trainer class for GNN Link Prediction
    This class is used to train the GNN model for the link prediction task
    The model is trained to predict the link between two nodes
    """
    def __init__(
            self, 
            model: GNNConv, 
            predictor: Union[EdgeClassifer, NodeClassifier], 
            cls_label,
            lr=1e-3,
            num_epochs=100,
            use_edge_attrs=False
        ) -> None:
        self.model = model
        self.predictor = predictor
        self.model.to(device)
        self.predictor.to(device)

        self.cls_label = cls_label
        self.num_epochs = num_epochs
        
        self.optimizer = Adam(chain(model.parameters(), predictor.parameters()), lr=lr)
        self.scheduler = CosineAnnealingLR(self.optimizer, T_max=num_epochs)
        
        self.edge2index = lambda g: torch.stack(list(g.edges())).contiguous()
        self.results = list()
        self.criterion = nn.CrossEntropyLoss()

        self.use_edge_attrs = use_edge_attrs

        print("GNN Trainer initialized.")


    @abstractmethod
    def train(self):
        pass

    @abstractmethod
    def test(self):
        pass


    def get_logits(self, x, edge_index, edge_attr=None):
        edge_index = edge_index.to(device)
        x = x.to(device)

        if edge_attr is not None:
            edge_attr = edge_attr.to(device)
            h = self.model(x, edge_index, edge_attr)
        else:
            h = self.model(x, edge_index)
        return h
    

    def get_prediction_score(self, h, edge_index=None, edge_attr=None):
        h = h.to(device)
        if edge_attr is not None:
            edge_attr = edge_attr.to(device)
            edge_index = edge_index.to(device)
            prediction_score = self.predictor(h, edge_index, edge_attr)
        elif edge_index is not None:
            edge_index = edge_index.to(device)
            prediction_score = self.predictor(h, edge_index)
        else:
            prediction_score = self.predictor(h)
        return prediction_score
        

    def compute_metrics(self, predictions: torch.Tensor, labels: torch.Tensor, multi_class=True):
        
        scores = torch.argmax(predictions, dim=-1)
        if multi_class:
            f1 = f1_score(labels.numpy(), scores.numpy(), average='weighted')

        else:
            roc_auc = roc_auc_score(labels.numpy(), scores.numpy())
            f1 = f1_score(labels.numpy(), scores.numpy())
            
        accuracy = accuracy_score(labels.numpy(), scores.numpy())
        balanced_accuracy = balanced_accuracy_score(labels.numpy(), scores.numpy())


        return {
            'roc_auc': roc_auc,
            'f1-score': f1,
            'accuracy': accuracy,
            'balanced_accuracy': balanced_accuracy
        }
    
    
    def plot_metrics(self):
        results = pd.DataFrame(self.results)
        df = pd.DataFrame(results, index=range(1, len(results)+1))
        df['epoch'] = df.index

        columns = [c for c in df.columns if c not in ['epoch', 'phase']]
        df.loc[df['phase'] == 'test'].plot(x='epoch', y=columns, kind='line')


    def run(self):
        for _ in tqdm(range(self.num_epochs), desc="Running Epochs"):
            self.train()
            self.test()