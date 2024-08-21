from torch_geometric.loader import DataLoader
from torch_geometric.data import Data
import torch
from collections import defaultdict
from typing import List
from sklearn.metrics import (
    balanced_accuracy_score,
    f1_score, 
    recall_score, 
    roc_auc_score,
    accuracy_score
)

from models.gnn_layers import (
    GNNConv, 
    EdgeClassifer
)

from trainers.gnn_trainer import Trainer

from utils import get_device
device = get_device()


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
        ) -> None:

        super().__init__(
            model=model,
            predictor=predictor,
            lr=lr,
            cls_label=cls_label,
            num_epochs=num_epochs
        )        
        self.dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        self.results = list()

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
            
            h = self.get_logits(data.x, data.train_pos_edge_label_index)

            pos_score = self.get_prediction_score(data.train_pos_edge_label_index, h)
            neg_score = self.get_prediction_score(data.train_neg_edge_label_index, h)
            loss = self.compute_loss(pos_score, neg_score)
            all_labels.append(torch.cat([torch.ones(pos_score.size(0)), torch.zeros(neg_score.size(0))]))
            all_preds.append(torch.cat([pos_score, neg_score]))

            loss.backward()
            self.optimizer.step()
            self.scheduler.step()
            epoch_loss += loss.item()
                        
        
        all_preds = torch.cat(all_preds, dim=0)
        all_labels = torch.cat(all_labels, dim=0)
        epoch_metrics = self.compute_metrics(all_preds, all_labels)
        epoch_metrics['loss'] = epoch_loss        
        epoch_metrics['phase'] = 'train'


    def test(self):
        self.model.eval()
        self.predictor.eval()
        all_preds, all_labels = list(), list()
        with torch.no_grad():
            epoch_loss = 0
            epoch_metrics = defaultdict(float)
            for data in self.dataloader:
                h = self.get_logits(data.x, data.test_pos_edge_label_index)
                pos_score = self.get_prediction_score(data.test_pos_edge_label_index, h)
                neg_score = self.get_prediction_score(data.test_neg_edge_label_index, h)
                loss = self.compute_loss(pos_score, neg_score)
                all_labels.append(torch.cat([torch.ones(pos_score.size(0)), torch.zeros(neg_score.size(0))]))
                all_preds.append(torch.cat([pos_score, neg_score]))

                epoch_loss += loss.item()

            all_preds = torch.cat(all_preds, dim=0)
            all_labels = torch.cat(all_labels, dim=0)
            epoch_metrics = self.compute_metrics(all_preds, all_labels)

            epoch_metrics['loss'] = epoch_loss
            epoch_metrics['phase'] = 'test'
            # print(f"Epoch Test Loss: {epoch_loss}\nTest Accuracy: {epoch_acc}\nTest F1: {epoch_f1}")
            self.results.append(epoch_metrics)

            print(f"Epoch: {len(self.results)}\n{epoch_metrics}")
            

    def compute_loss(self, pos_score, neg_score):
        pos_label = torch.ones(pos_score.size(0), dtype=torch.long).to(device)
        neg_label = torch.zeros(neg_score.size(0), dtype=torch.long).to(device)

        scores = torch.cat([pos_score, neg_score], dim=0)
        labels = torch.cat([pos_label, neg_label], dim=0)

        loss = self.criterion(scores, labels)
        return loss

    
    def compute_metrics(self, pos_score, neg_score):
        pos_label = torch.ones(pos_score.size(0), dtype=torch.long).to(device)
        neg_label = torch.zeros(neg_score.size(0), dtype=torch.long).to(device)

        scores = torch.cat([pos_score, neg_score], dim=0)
        labels = torch.cat([pos_label, neg_label], dim=0)

        scores = torch.argmax(scores, dim=-1)

        roc_auc = roc_auc_score(labels.cpu().numpy(), scores.cpu().numpy())
        f1 = f1_score(labels.cpu().numpy(), scores.cpu().numpy())
        recall = recall_score(labels.cpu().numpy(), scores.cpu().numpy())
        accuracy = accuracy_score(labels.cpu().numpy(), scores.cpu().numpy())
        balanced_accuracy = balanced_accuracy_score(labels.cpu().numpy(), scores.cpu().numpy())


        return {
            'roc_auc': roc_auc,
            'f1-score': f1,
            'accuracy': accuracy,
            'recall': recall,
            'balanced_accuracy': balanced_accuracy
        }