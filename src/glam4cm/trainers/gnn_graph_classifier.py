from typing import List, Tuple
import torch
from collections import defaultdict
from torch_geometric.loader import DataLoader

from torch_geometric.data import Data
from glam4cm.models.gnn_layers import (
    GNNConv,
    GraphClassifer
)
from glam4cm.trainers.gnn_trainer import Trainer
from glam4cm.settings import device


class GNNGraphClassificationTrainer(Trainer):
    """
    Trainer class for GNN Graph Classfication
    This class is used to train the GNN model for the link prediction task
    The model is trained to predict the link between two nodes
    """
    def __init__(
            self, 
            model: GNNConv, 
            predictor: GraphClassifer,
            dataset: List[Tuple[Data, Data]],
            cls_label='label',
            lr=1e-4,
            num_epochs=100,
            batch_size=32,
            use_edge_attrs=False,
            logs_dir='./logs'
        ) -> None:

        super().__init__(
            model=model,
            predictor=predictor,
            cls_label='type',
            lr=lr,
            num_epochs=num_epochs,
            use_edge_attrs=use_edge_attrs,
            logs_dir=logs_dir
        )

        self.cls_label = cls_label
        self.dataloaders = dict()
        self.dataloaders['train'] = DataLoader(dataset['train'], batch_size=batch_size, shuffle=True)
        self.dataloaders['test'] = DataLoader(dataset['test'], batch_size=batch_size, shuffle=False)

        self.results = list()

        print("GNN Trainer initialized.")


    def train(self):
        self.model.train()
        self.predictor.train()

        epoch_loss = 0
        epoch_metrics = defaultdict(float)
        preds, all_labels = list(), list()
        for data in self.dataloaders['train']:
            self.optimizer.zero_grad()
            self.model.train()
            self.predictor.train()
            
            h = self.model(data.x.to(device), data.edge_index.to(device))
            g_pred = self.predictor(h, data.batch.to(device))

            
            labels = getattr(data, f"graph_{self.cls_label}")
            loss = self.criterion(g_pred, labels.to(device))

            preds.append(g_pred.detach().cpu())
            all_labels.append(labels)

            loss.backward()
            self.optimizer.step()
            self.scheduler.step()
            epoch_loss += loss.item()

        
        preds = torch.cat(preds, dim=0)
        labels = torch.cat(all_labels, dim=0)
        
        epoch_metrics = self.compute_metrics(preds, labels)
        epoch_metrics['loss'] = epoch_loss
        epoch_metrics['phase'] = 'train'

        self.results.append(epoch_metrics)

        return epoch_metrics


    def test(self):
        self.model.eval()
        self.predictor.eval()
        with torch.no_grad():
            epoch_loss = 0
            preds, all_labels = list(), list()
            for data in self.dataloaders['test']:
                h = self.model(data.x.to(device), data.edge_index.to(device))
                g_pred = self.predictor(h, data.batch.to(device))
                labels = getattr(data, f"graph_{self.cls_label}")

                loss = self.criterion(g_pred, labels.to(device))
                epoch_loss += loss.item()

                preds.append(g_pred.cpu().detach())
                all_labels.append(labels.cpu())

            
            preds = torch.cat(preds, dim=0)
            labels = torch.cat(all_labels, dim=0)

            epoch_metrics = self.compute_metrics(preds, labels)
            epoch_metrics['loss'] = epoch_loss
            epoch_metrics['phase'] = 'test'
            self.results.append(epoch_metrics)
        
        s2t = lambda x: x.replace("_", " ").title()
        print(f"Epoch: {len(self.results)//2} {' | '.join([f'{s2t(k)}: {v:.4f}' for k, v in epoch_metrics.items() if k != 'phase'])}")

        return epoch_metrics