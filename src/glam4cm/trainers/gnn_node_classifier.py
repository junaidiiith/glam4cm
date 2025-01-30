from typing import List
from torch_geometric.loader import DataLoader
import torch
from collections import defaultdict
from torch_geometric.data import Data
from glam4cm.models.gnn_layers import (
    GNNConv, 
    NodeClassifier
)
from glam4cm.trainers.gnn_trainer import Trainer
from glam4cm.settings import device


class GNNNodeClassificationTrainer(Trainer):
    """
    Trainer class for GNN Link Prediction
    This class is used to train the GNN model for the link prediction task
    The model is trained to predict the link between two nodes
    """
    def __init__(
            self, 
            model: GNNConv, 
            predictor: NodeClassifier, 
            dataset: List[Data],
            cls_label,
            exclude_labels=None,
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

        self.exclude_labels = torch.tensor(exclude_labels, dtype=torch.long)
        self.results = list()
        self.dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        print("GNN Trainer initialized.")



    def train(self):
        self.model.train()
        self.predictor.train()

        all_preds, all_labels = list(), list()
        epoch_loss = 0
        epoch_metrics = defaultdict(float)
        # for i, data in tqdm(enumerate(self.dataloader), desc=f"Training batches", total=len(self.dataloader)):
        for data in self.dataloader:
            self.optimizer.zero_grad()
            self.model.zero_grad()
            self.predictor.zero_grad()
            
            h = self.get_logits(
                data.x, 
                data.edge_index,
                data.edge_attr if self.use_edge_attrs else None
            )
            scores = self.get_prediction_score(h)[data.train_node_mask]
            labels = getattr(data, f"node_{self.cls_label}")[data.train_node_mask]
            
            mask = ~torch.isin(labels, self.exclude_labels)
            labels = labels[mask]
            scores = scores[mask]

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

        return epoch_metrics


    def test(self):
        self.model.eval()
        self.predictor.eval()
        all_preds, all_labels = list(), list()
        with torch.no_grad():
            epoch_loss = 0
            epoch_metrics = defaultdict(float)
            # for _, data in tqdm(enumerate(self.dataloader), desc=f"Evaluating batches", total=len(self.dataloader)):
            for data in self.dataloader:
                h = self.get_logits(
                    data.x, 
                    data.edge_index,
                    data.edge_attr if self.use_edge_attrs else None
                )
                
                scores = self.get_prediction_score(h)[data.test_node_mask]
                labels = getattr(data, f"node_{self.cls_label}")[data.test_node_mask]

                mask = ~torch.isin(labels, self.exclude_labels)
                labels = labels[mask]
                scores = scores[mask]
                loss = self.criterion(scores, labels.to(device))
                epoch_loss += loss.item()


                all_preds.append(scores.detach().cpu())
                all_labels.append(labels)
                

            all_preds = torch.cat(all_preds, dim=0)
            all_labels = torch.cat(all_labels, dim=0)
            epoch_metrics = self.compute_metrics(all_preds, all_labels)
            
            epoch_metrics['loss'] = epoch_loss
            epoch_metrics['phase'] = 'test'
            self.results.append(epoch_metrics)

            print(f"Epoch: {len(self.results)}\n{epoch_metrics}")

        return epoch_metrics