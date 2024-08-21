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

        print("GNN Trainer initialized.")


    @abstractmethod
    def train(self):
        pass

    @abstractmethod
    def test(self):
        pass


    def get_logits(self, x, edge_index):
        edge_index = edge_index.to(device)
        x = x.to(device)
        h = self.model(x, edge_index)
        return h
    

    def get_prediction_score(self, edge_index, h):
        h = h.to(device)
        edge_index = edge_index.to(device)
        prediction_score = self.predictor(h, edge_index)
        return prediction_score
    

    def compute_loss(self, scores, labels):
        loss = self.criterion(scores, labels.to(device))
        return loss
    
    
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