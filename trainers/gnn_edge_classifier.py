from random import shuffle
from torch_geometric.loader import DataLoader
import torch
from collections import defaultdict
from typing import List
import pandas as pd
from sklearn.metrics import (
    balanced_accuracy_score,
    f1_score, 
    recall_score, 
    roc_auc_score,
    accuracy_score
)

from data_loading.graph_dataset import GraphEdgeDataset
from models.gnn_layers import (
    GNNConv, 
    EdgeClassifer
)
from utils import get_device, randomize_features
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
            predictor: EdgeClassifer, 
            dataset: List[GraphEdgeDataset],
            lr=1e-3,
            num_epochs=100,
            batch_size=32,
            randomize_ne = False
        ) -> None:
        self.model = model
        self.predictor = predictor
        self.model.to(device)
        self.predictor.to(device)
        
        dataset = [g.data for g in dataset]
        shuffle(dataset)

        if randomize_ne:
            dataset = randomize_features(dataset, 768)

        self.dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

        self.optimizer = Adam(chain(model.parameters(), predictor.parameters()), lr=lr)
        self.scheduler = CosineAnnealingLR(self.optimizer, T_max=num_epochs)
        
        self.edge2index = lambda g: torch.stack(list(g.edges())).contiguous()
        self.results = list()
        self.criterion = nn.CrossEntropyLoss()

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
            
            h = self.get_logits(data.x, data.train_pos_edge_label_index)

            scores = self.get_prediction_score(data.train_pos_edge_label_index, h)
            labels = data.edge_classes[data.train_edge_idx]
            loss = self.compute_loss(scores, labels)
            all_preds.append(scores)
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
                h = self.get_logits(data.x, data.test_pos_edge_label_index)

                scores = self.get_prediction_score(data.test_pos_edge_label_index, h)
                labels = data.edge_classes[data.test_edge_idx]
                all_preds.append(scores)
                all_labels.append(labels)
                loss = self.compute_loss(scores, labels)
                
                epoch_loss += loss.item()

            all_preds = torch.cat(all_preds, dim=0)
            all_labels = torch.cat(all_labels, dim=0)
            epoch_metrics = self.compute_metrics(all_preds, all_labels)
            
            epoch_metrics['loss'] = epoch_loss
            epoch_metrics['phase'] = 'test'
            # print(f"Epoch Test Loss: {epoch_loss}\nTest Accuracy: {epoch_acc}\nTest F1: {epoch_f1}")
            self.results.append(epoch_metrics)

            print(f"Epoch: {len(self.results)}\n{epoch_metrics}")
            

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
    

    def compute_metrics(self, scores, labels):
        preds = torch.argmax(scores, dim=-1)
        roc_auc = roc_auc_score(labels.cpu().numpy(), scores.cpu().numpy(), multi_class='ovr')
        f1 = f1_score(labels.cpu().numpy(), preds.cpu().numpy(), average='weighted')
        accuracy = accuracy_score(labels.cpu().numpy(), preds.cpu().numpy())
        recall = recall_score(labels.cpu().numpy(), preds.cpu().numpy(), average='weighted')

        balanced_accuracy = balanced_accuracy_score(labels.cpu().numpy(), preds.cpu().numpy())

        return {
            'roc_auc': roc_auc,
            'f1-score': f1,
            'balanced_accuracy': balanced_accuracy,
            'recall': recall,
            'accuracy': accuracy,
        }
    
    def plot_metrics(self):
        results = pd.DataFrame(self.results)
        df = pd.DataFrame(results, index=range(1, len(results)+1))
        df['epoch'] = df.index

        columns = [c for c in df.columns if c not in ['epoch', 'phase']]
        df.loc[df['phase'] == 'test'].plot(x='epoch', y=columns, kind='line')


    def run_epochs(self, num_epochs):
        for _ in tqdm(range(num_epochs), desc="Running Epochs"):
            self.train()
            self.test()
        
