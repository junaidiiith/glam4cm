from typing import List
import torch
from collections import defaultdict
from torch_geometric.loader import DataLoader
from glam4cm.data_loading.data import GraphData
from glam4cm.models.gnn_layers import (
    GNNConv, 
    EdgeClassifer
)

from glam4cm.trainers.gnn_trainer import Trainer
from glam4cm.settings import device


class GNNEdgeClassificationTrainer(Trainer):
    """
    Trainer class for GNN Link Prediction
    This class is used to train the GNN model for the link prediction task
    The model is trained to predict the link between two nodes
    """
    def __init__(
            self, 
            model: GNNConv, 
            predictor: EdgeClassifer, 
            dataset: List[GraphData],
            cls_label='type',
            lr=1e-3,
            num_epochs=100,
            batch_size=32,
            use_edge_attrs=False,
            logs_dir='./logs'
        ) -> None:

        super().__init__(
            model=model,
            predictor=predictor,
            cls_label=cls_label,
            lr=lr,
            num_epochs=num_epochs,
            use_edge_attrs=use_edge_attrs,
            logs_dir=logs_dir
        )

        self.dataloader = DataLoader(
            dataset, 
            batch_size=batch_size, 
            shuffle=True
        )

        print("GNN Trainer initialized.")


    def train(self):
        self.model.train()
        self.predictor.train()

        all_preds, all_labels = list(), list()
        epoch_loss = 0
        epoch_metrics = defaultdict(float)
        for data in self.dataloader:
            self.optimizer.zero_grad()
            self.model.zero_grad()
            self.predictor.zero_grad()
            x = data.x
            edge_index =  data.train_pos_edge_label_index
            train_mask = data.train_edge_mask
            edge_attr = data.edge_attr[train_mask] if self.use_edge_attrs else None
            
            h = self.get_logits(x, edge_index, edge_attr)
            scores = self.get_prediction_score(h, edge_index, edge_attr)
            labels = getattr(data, f"edge_{self.cls_label}")[train_mask]
            loss = self.criterion(scores, labels.to(device))
            all_preds.append(scores.detach().cpu())
            all_labels.append(labels)

            loss.backward()
            self.optimizer.step()
            self.scheduler.step()
            epoch_loss += loss.item()
                        
        
        all_preds = torch.cat(all_preds, dim=0)
        all_labels = torch.cat(all_labels, dim=0)
        epoch_metrics = self.compute_metrics(all_preds, all_labels)
        epoch_metrics['loss'] = epoch_loss        
        epoch_metrics['phase'] = 'train'

        # print(f"Train Metrics: {epoch_metrics}")

        return epoch_metrics


    def test(self):
        self.model.eval()
        self.predictor.eval()
        all_preds, all_labels = list(), list()
        with torch.no_grad():
            epoch_loss = 0
            epoch_metrics = defaultdict(float)
            for data in self.dataloader:
                x = data.x
                edge_index =  data.test_pos_edge_label_index
                test_mask = data.test_edge_mask
                edge_attr = data.edge_attr[test_mask] if self.use_edge_attrs else None
                
                h = self.get_logits(x, edge_index, edge_attr)
                scores = self.get_prediction_score(h, edge_index, edge_attr)
                labels = getattr(data, f"edge_{self.cls_label}")[test_mask]
                all_preds.append(scores.detach().cpu())
                all_labels.append(labels)
                loss = self.criterion(scores, labels.to(device))
                
                epoch_loss += loss.item()

            all_preds = torch.cat(all_preds, dim=0)
            all_labels = torch.cat(all_labels, dim=0)
            epoch_metrics = self.compute_metrics(all_preds, all_labels)
            
            epoch_metrics['loss'] = epoch_loss
            epoch_metrics['phase'] = 'test'
            # print(f"Epoch Test Loss: {epoch_loss}\nTest Accuracy: {epoch_acc}\nTest F1: {epoch_f1}")
            self.results.append(epoch_metrics)

            print(f"Epoch: {len(self.results)}\n{epoch_metrics}")
        
        return epoch_metrics