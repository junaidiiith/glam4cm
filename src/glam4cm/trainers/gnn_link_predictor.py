from torch_geometric.loader import DataLoader
from torch_geometric.data import Data
import torch
from collections import defaultdict
from typing import List

from glam4cm.models.gnn_layers import (
    GNNConv, 
    EdgeClassifer
)

from glam4cm.trainers.gnn_trainer import Trainer
from tqdm.auto import tqdm
from glam4cm.settings import device


class GNNLinkPredictionTrainer(Trainer):
    """
    Trainer class for GNN Link Prediction
    This class is used to train the GNN model for the link prediction task
    The model is trained to predict the link between two nodes
    """
    def __init__(
            self, 
            model: GNNConv, 
            predictor: EdgeClassifer, 
            dataset: List[Data],
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
            lr=lr,
            cls_label=cls_label,
            num_epochs=num_epochs,
            use_edge_attrs=use_edge_attrs,
            logs_dir=logs_dir
        )        
        self.dataloader = DataLoader(
            dataset, batch_size=batch_size, shuffle=True
        )
        self.results = list()

        print("GNN Trainer initialized.")



    def train(self):
        self.model.train()
        self.predictor.train()

        all_preds, all_labels = list(), list()
        epoch_loss = 0
        epoch_metrics = defaultdict(float)
        for data in tqdm(self.dataloader, desc='Training Batches'):
            self.optimizer.zero_grad()
            self.model.zero_grad()
            self.predictor.zero_grad()

            x = data.x
            pos_edge_index =  data.train_pos_edge_label_index
            neg_edge_index = data.train_neg_edge_label_index
            train_mask = data.train_edge_mask
            edge_attr = data.edge_attr[train_mask] if self.use_edge_attrs else None
            
            h = self.get_logits(x, pos_edge_index, edge_attr)
            # h = x

            pos_scores = self.get_prediction_score(h, pos_edge_index, edge_attr)
            neg_scores = self.get_prediction_score(h, neg_edge_index, edge_attr)
            loss = self.compute_loss(pos_scores, neg_scores)
            all_labels.append(torch.cat([torch.ones(pos_scores.size(0)), torch.zeros(neg_scores.size(0))]))
            all_preds.append(torch.cat([pos_scores.detach().cpu(), neg_scores.detach().cpu()]))

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
            for data in tqdm(self.dataloader, desc='Testing Batches'):
                
                x = data.x
                pos_edge_index =  data.test_pos_edge_label_index
                neg_edge_index = data.test_neg_edge_label_index
                test_mask = data.test_edge_mask
                edge_attr = data.edge_attr[test_mask] if self.use_edge_attrs else None


                h = self.get_logits(x, pos_edge_index, edge_attr)
                # h = x
                pos_score = self.get_prediction_score(h, pos_edge_index, edge_attr)
                neg_score = self.get_prediction_score(h, neg_edge_index, edge_attr)

                loss = self.compute_loss(pos_score, neg_score)
                all_labels.append(torch.cat([torch.ones(pos_score.size(0)), torch.zeros(neg_score.size(0))]))
                all_preds.append(torch.cat([pos_score.detach().cpu(), neg_score.detach().cpu()]))

                epoch_loss += loss.item()

            all_preds = torch.cat(all_preds, dim=0)
            all_labels = torch.cat(all_labels, dim=0)
            epoch_metrics = self.compute_metrics(all_preds, all_labels)

            epoch_metrics['loss'] = epoch_loss
            epoch_metrics['phase'] = 'test'
            # print(f"Epoch Test Loss: {epoch_loss}\nTest Accuracy: {epoch_acc}\nTest F1: {epoch_f1}")
            self.results.append(epoch_metrics)

            print(f"Test Epoch: {len(self.results)}\n{epoch_metrics}")

        return epoch_metrics        

    def compute_loss(self, pos_score, neg_score):
        pos_label = torch.ones(pos_score.size(0), dtype=torch.long).to(device)
        neg_label = torch.zeros(neg_score.size(0), dtype=torch.long).to(device)

        scores = torch.cat([pos_score, neg_score], dim=0)
        labels = torch.cat([pos_label, neg_label], dim=0)

        loss = self.criterion(scores, labels)
        return loss