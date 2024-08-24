from typing import List
from torch_geometric.loader import DataLoader
import torch
from collections import defaultdict
from sklearn.metrics import (
    balanced_accuracy_score,
    f1_score, 
    recall_score, 
    accuracy_score
)

from torch_geometric.data import Data
from models.gnn_layers import (
    GNNConv, 
    NodeClassifier
)
from utils import get_device
from trainers.gnn_trainer import Trainer

device = get_device()


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
            lr=1e-3,
            num_epochs=100,
            batch_size=32,
        ) -> None:

        super().__init__(
            model=model,
            predictor=predictor,
            cls_label=cls_label,
            lr=lr,
            num_epochs=num_epochs
        )

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
            
            h = self.get_logits(data.x, data.edge_index)
            scores = self.get_prediction_score(h)[data.train_node_idx]
            labels = getattr(data, f"node_{self.cls_label}")[data.train_node_idx]
            loss = self.compute_loss(scores, labels)
            
            all_preds.append(scores.detach())
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


    def test(self):
        self.model.eval()
        self.predictor.eval()
        all_preds, all_labels = list(), list()
        with torch.no_grad():
            epoch_loss = 0
            epoch_metrics = defaultdict(float)
            # for _, data in tqdm(enumerate(self.dataloader), desc=f"Evaluating batches", total=len(self.dataloader)):
            for data in self.dataloader:
                h = self.get_logits(data.x, data.edge_index)
                scores = self.get_prediction_score(h)[data.test_node_idx]
                labels = getattr(data, f"node_{self.cls_label}")[data.test_node_idx]

                loss = self.compute_loss(scores, labels)
                epoch_loss += loss.item()


                all_preds.append(scores.detach())
                all_labels.append(labels)
                
                
                

            all_preds = torch.cat(all_preds, dim=0)
            all_labels = torch.cat(all_labels, dim=0)
            epoch_metrics = self.compute_metrics(all_preds, all_labels)
            
            epoch_metrics['loss'] = epoch_loss
            epoch_metrics['phase'] = 'test'
            # print(f"Epoch Test Loss: {epoch_loss}\nTest Accuracy: {epoch_acc}\nTest F1: {epoch_f1}")
            self.results.append(epoch_metrics)

            print(f"Epoch: {len(self.results)}\n{epoch_metrics}")

    

    def compute_metrics(self, scores, labels):
        preds = torch.argmax(scores, dim=-1)
        f1 = f1_score(labels.cpu().numpy(), preds.cpu().numpy(), average='weighted')
        accuracy = accuracy_score(labels.cpu().numpy(), preds.cpu().numpy())
        recall = recall_score(labels.cpu().numpy(), preds.cpu().numpy(), average='weighted')

        balanced_accuracy = balanced_accuracy_score(labels.cpu().numpy(), preds.cpu().numpy())

        return {
            'f1-score': f1,
            'balanced_accuracy': balanced_accuracy,
            'recall': recall,
            'accuracy': accuracy,
        }