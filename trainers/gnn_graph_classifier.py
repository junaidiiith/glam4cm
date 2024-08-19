import torch
from collections import defaultdict
from typing import List
import pandas as pd
from sklearn.metrics import (
    balanced_accuracy_score,
    f1_score, 
    recall_score, 
    accuracy_score
)

from data_loading.graph_dataset import GraphEdgeDataset
from models.gnn_layers import (
    GNNConv,
    GraphClassifer
)
from utils import get_device, randomize_features
from itertools import chain
from tqdm.auto import tqdm
import torch.nn as nn
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.optim import Adam

device = get_device()


from torch_geometric.loader import DataLoader
from random import shuffle


class Trainer:
    """
    Trainer class for GNN Graph Classfication
    This class is used to train the GNN model for the link prediction task
    The model is trained to predict the link between two nodes
    """
    def __init__(
            self, 
            gnn_conv: GNNConv, 
            classifier: GraphClassifer,
            dataset: GraphEdgeDataset,
            tr=0.2,
            lr=1e-4,
            num_epochs=100,
            batch_size=32,
            randomize_ne = False
        ) -> None:
        tr = 1 - tr
        self.gnn_conv = gnn_conv
        self.classifier = classifier

        self.gnn_conv.to(device)
        self.classifier.to(device)

        self.num_epochs = num_epochs

        self.dataloaders = dict()
        dataset = [g.data for g in dataset]
        shuffle(dataset)
        if randomize_ne:
            dataset = randomize_features(dataset, 768)

        train_dataset = dataset[:int(tr * len(dataset))]
        test_dataset = dataset[int(tr * len(dataset)):]
        self.dataloaders['train'] = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        self.dataloaders['test'] = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

        self.optimizer = Adam(chain(self.gnn_conv.parameters(), self.classifier.parameters()), lr=lr)
        self.scheduler = CosineAnnealingLR(self.optimizer, T_max=num_epochs)
        
        self.edge2index = lambda g: torch.stack(list(g.edges())).contiguous()
        self.results = list()
        self.criterion = nn.CrossEntropyLoss()

        print("GNN Trainer initialized.")



    def train(self):
        self.gnn_conv.train()
        self.classifier.train()

        epoch_loss = 0
        epoch_metrics = defaultdict(float)
        preds, labels = list(), list()
        # for i, data in tqdm(enumerate(self.dataloader), desc=f"Training batches", total=len(self.dataloader)):
        for data in self.dataloaders['train']:
            self.optimizer.zero_grad()
            self.gnn_conv.train()
            self.classifier.train()
            
            h = self.gnn_conv(data.x.to(device), data.edge_index.to(device))
            g_pred = self.classifier(h, data.batch.to(device))

            preds.append(g_pred.cpu().detach())
            labels.append(data.y.cpu().detach())
            
            loss = self.criterion(g_pred, data.y.to(device))

            loss.backward()
            self.optimizer.step()
            self.scheduler.step()
            epoch_loss += loss.item()
                        
        
        preds = torch.cat(preds, dim=0)
        labels = torch.cat(labels, dim=0)
        
        epoch_metrics = self.compute_metrics(preds, labels)
        epoch_metrics['loss'] = epoch_loss
        epoch_metrics['phase'] = 'train'
        self.results.append(epoch_metrics)


    def test(self):
        self.gnn_conv.eval()
        self.classifier.eval()
        with torch.no_grad():
            epoch_loss = 0
            preds, labels = list(), list()
            for data in self.dataloaders['test']:
                h = self.gnn_conv(data.x.to(device), data.edge_index.to(device))
                g_pred = self.classifier(h, data.batch.to(device))

                preds.append(g_pred.cpu().detach())
                labels.append(data.y.cpu().detach())

                loss = self.criterion(g_pred, data.y.to(device))
                epoch_loss += loss.item()

            
            preds = torch.cat(preds, dim=0)
            labels = torch.cat(labels, dim=0)

            epoch_metrics = self.compute_metrics(preds, labels)
            epoch_metrics['loss'] = epoch_loss
            epoch_metrics['phase'] = 'test'
            self.results.append(epoch_metrics)
        
        print(f"Epoch: {len(self.results)//2} | Loss: {epoch_loss} | F1: {epoch_metrics['f1-score']} | Acc: {epoch_metrics['accuracy']} | Balanced Acc: {epoch_metrics['balanced_accuracy']}")
            

    def get_conv(self, x, edge_index):
        edge_index = edge_index.to(device)
        x = x.to(device)
        h = self.model(x, edge_index)
        return h
    

    def get_classification(self, h, batch):
        h = h.to(device)
        batch = batch.to(device)
        y_pred = self.classifier(h, batch)
        return y_pred
    

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