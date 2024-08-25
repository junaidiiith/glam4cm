from typing import List, Tuple
from sklearn.metrics import (
    balanced_accuracy_score,
    f1_score, 
    recall_score, 
    accuracy_score
)
import torch
from collections import defaultdict
from torch_geometric.loader import DataLoader

from torch_geometric.data import Data
from models.gnn_layers import (
    GNNConv,
    GraphClassifer
)
from utils import get_device
from trainers.gnn_trainer import Trainer

device = get_device()


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
            use_edge_attrs=False
        ) -> None:

        super().__init__(
            model=model,
            predictor=predictor,
            cls_label='type',
            lr=lr,
            num_epochs=num_epochs,
            use_edge_attrs=use_edge_attrs
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
        # for i, data in tqdm(enumerate(self.dataloader), desc=f"Training batches", total=len(self.dataloader)):
        for data in self.dataloaders['train']:
            self.optimizer.zero_grad()
            self.model.train()
            self.predictor.train()
            
            h = self.model(data.x.to(device), data.edge_index.to(device))
            g_pred = self.predictor(h, data.batch.to(device))

            
            labels = getattr(data, f"graph_{self.cls_label}")
            loss = self.criterion(g_pred, labels.to(device))

            preds.append(g_pred.cpu().detach())
            all_labels.append(labels.cpu())

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

                preds.append(g_pred.cpu().detach())
                all_labels.append(labels.cpu())
            

                loss = self.criterion(g_pred, labels.to(device))
                epoch_loss += loss.item()

            
            preds = torch.cat(preds, dim=0)
            labels = torch.cat(all_labels, dim=0)

            epoch_metrics = self.compute_metrics(preds, labels)
            epoch_metrics['loss'] = epoch_loss
            epoch_metrics['phase'] = 'test'
            self.results.append(epoch_metrics)
        
        print(f"Epoch: {len(self.results)//2} | Loss: {epoch_loss} | F1: {epoch_metrics['f1-score']} | Acc: {epoch_metrics['accuracy']} | Balanced Acc: {epoch_metrics['balanced_accuracy']}")
            

    def compute_metrics(self, scores, labels):
        preds = scores.argmax(dim=-1).cpu().numpy()
        labels = labels.cpu().numpy()
        scores = scores.cpu().numpy()
        metrics = {
            'f1-score': f1_score(labels, preds, average='macro'),
            'accuracy': accuracy_score(labels, preds),
            'recall': recall_score(labels, preds, average='macro'),
            'balanced_accuracy': balanced_accuracy_score(labels, preds)
        }
        
        return metrics