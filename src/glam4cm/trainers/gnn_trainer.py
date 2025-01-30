from abc import abstractmethod
import torch
from typing import Union
import pandas as pd

from glam4cm.models.gnn_layers import (
    GNNConv, 
    EdgeClassifer,
    NodeClassifier
)
from glam4cm.settings import device
from itertools import chain
from tqdm.auto import tqdm
import torch.nn as nn
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.optim import Adam
from torch_geometric.loader import DataLoader

from tensorboardX import SummaryWriter
from glam4cm.trainers.metrics import compute_classification_metrics


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
            use_edge_attrs=False,

            logs_dir='./logs'
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

        self.logs_dir = logs_dir

        self.writer = SummaryWriter(log_dir=self.logs_dir)

        print("GNN Trainer initialized.")


    def set_dataloader(self, dataset, batch_size):
        self.dataloader = DataLoader(
            dataset, batch_size=batch_size, shuffle=True
        )

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
        
    
    def plot_metrics(self):
        results = pd.DataFrame(self.results)
        df = pd.DataFrame(results, index=range(1, len(results)+1))
        df['epoch'] = df.index

        columns = [c for c in df.columns if c not in ['epoch', 'phase']]
        df.loc[df['phase'] == 'test'].plot(x='epoch', y=columns, kind='line')


    def run(self):
        for epoch in tqdm(range(self.num_epochs), desc="Running Epochs"):
            train_metrics = self.train()
            test_metrics = self.test()

            for k, v in train_metrics.items():
                if k != 'phase':
                    self.writer.add_scalar(f"train/{k}", v, epoch)
            
            for k, v in test_metrics.items():
                if k != 'phase':
                    self.writer.add_scalar(f"test/{k}", v, epoch)
    
        self.writer.close()
    
    def compute_metrics(self, all_preds, all_labels):
        return compute_classification_metrics(all_preds, all_labels)