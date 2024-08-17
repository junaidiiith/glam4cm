import torch
from collections import defaultdict
from typing import List
import pandas as pd
from sklearn.metrics import (
    f1_score, 
    recall_score, 
    roc_auc_score,
    accuracy_score
)

from data_loading.data import LinkPredictionDataLoader
from models.gnn_layers import (
    GNNModel, 
    MLPPredictor
)
from settings import LP_TASK_EDGE_CLS
from utils import get_device
from itertools import chain
from torch_geometric.data import Data
from tqdm.auto import tqdm
import torch.nn as nn
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.optim import Adam

device = get_device()



class GNNLinkPredictionTrainer:
    """
    Trainer class for GNN Link Prediction
    This class is used to train the GNN model for the link prediction task
    The model is trained to predict the link between two nodes
    """
    def __init__(
            self, 
            model: GNNModel, 
            predictor: MLPPredictor, 
            dataset: List[Data],
            task_type=LP_TASK_EDGE_CLS,
            lr=1e-3,
            num_epochs=100,
            batch_size=32
        ) -> None:
        self.model = model
        self.predictor = predictor
        self.model.to(device)
        self.predictor.to(device)
        self.task = task_type

        self.dataloader = LinkPredictionDataLoader(
            dataset, 
            batch_size=batch_size, 
            shuffle=False
        )

        self.optimizer = Adam(chain(model.parameters(), predictor.parameters()), lr=lr)
        self.scheduler = CosineAnnealingLR(self.optimizer, T_max=num_epochs)
        
        self.edge2index = lambda g: torch.stack(list(g.edges())).contiguous()
        self.results = list()
        self.criterion = nn.CrossEntropyLoss()

        print("GNN Trainer initialized.")


    def train(self):
        self.model.train()
        self.predictor.train()

        epoch_loss = 0
        epoch_metrics = defaultdict(float)
        # for i, data in tqdm(enumerate(self.dataloader), desc=f"Training batches", total=len(self.dataloader)):
        for data in self.dataloader:
            self.optimizer.zero_grad()
            self.model.zero_grad()
            self.predictor.zero_grad()
            
            h = self.get_logits(data.x, data.train_pos_edge_label_index)

            if self.task == LP_TASK_EDGE_CLS:
                scores = self.get_prediction_score(data.train_pos_edge_label_index, h)
                labels = data.edge_classes[data.train_edge_idx]
                loss = self.compute_ec_loss(scores, labels)
                metrics = self.compute_ec_metrics(scores.detach(), labels)
            else:
                pos_score = self.get_prediction_score(data.train_pos_edge_label_index, h)
                neg_score = self.get_prediction_score(data.train_neg_edge_label_index, h)
                loss = self.compute_lp_loss(pos_score, neg_score)
                metrics = self.compute_lp_metrics(pos_score.detach(), neg_score.detach())

            loss.backward()
            self.optimizer.step()
            self.scheduler.step()
            epoch_loss += loss.item()
                
            for key, value in metrics.items():
                epoch_metrics[key] += value
        
        
        epoch_metrics['loss'] = epoch_loss
        
        for key in epoch_metrics:
            epoch_metrics[key] /= len(self.dataloader)
        
        epoch_metrics['phase'] = 'train'


    def test(self):
        self.model.eval()
        self.predictor.eval()
        with torch.no_grad():
            epoch_loss = 0
            epoch_metrics = defaultdict(float)
            # for _, data in tqdm(enumerate(self.dataloader), desc=f"Evaluating batches", total=len(self.dataloader)):
            for data in self.dataloader:
                h = self.get_logits(data.x, data.test_pos_edge_label_index)

                if self.task == LP_TASK_EDGE_CLS:
                    scores = self.get_prediction_score(data.test_pos_edge_label_index, h)
                    labels = data.edge_classes[data.test_edge_idx]
                    loss = self.compute_ec_loss(scores, labels)
                    metrics = self.compute_ec_metrics(scores.detach(), labels)
                else:
                    pos_score = self.get_prediction_score(data.test_pos_edge_label_index, h)
                    neg_score = self.get_prediction_score(data.test_neg_edge_label_index, h)
                    loss = self.compute_lp_loss(pos_score, neg_score)
                    metrics = self.compute_lp_metrics(pos_score.detach(), neg_score.detach())

                epoch_loss += loss.item()
                for key, value in metrics.items():
                    epoch_metrics[key] += value


            epoch_metrics['loss'] = epoch_loss
            
            for key in epoch_metrics:
                epoch_metrics[key] /= len(self.dataloader)
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
    

    def compute_lp_loss(self, pos_score, neg_score):
        pos_label = torch.ones(pos_score.size(0), dtype=torch.long).to(device)
        neg_label = torch.zeros(neg_score.size(0), dtype=torch.long).to(device)

        scores = torch.cat([pos_score, neg_score], dim=0)
        labels = torch.cat([pos_label, neg_label], dim=0)

        loss = self.criterion(scores, labels)
        return loss


    def compute_ec_loss(self, scores, labels):
        loss = self.criterion(scores, labels.to(device))
        return loss
    

    def compute_lp_metrics(self, pos_score, neg_score):
        pos_label = torch.ones(pos_score.size(0), dtype=torch.long).to(device)
        neg_label = torch.zeros(neg_score.size(0), dtype=torch.long).to(device)

        scores = torch.cat([pos_score, neg_score], dim=0)
        labels = torch.cat([pos_label, neg_label], dim=0)

        scores = torch.argmax(scores, dim=-1)

        roc_auc = roc_auc_score(labels.cpu().numpy(), scores.cpu().numpy())
        f1 = f1_score(labels.cpu().numpy(), scores.cpu().numpy())
        recall = recall_score(labels.cpu().numpy(), scores.cpu().numpy())
        accuracy = accuracy_score(labels.cpu().numpy(), scores.cpu().numpy())


        return {
            'roc_auc': roc_auc,
            'f1-score': f1,
            'accuracy': accuracy,
            'recall': recall
        }
    

    def compute_ec_metrics(self, scores, labels):
        score_probs = torch.nn.functional.softmax(scores, dim=-1)
        scores = torch.argmax(scores, dim=-1)
        roc_auc = roc_auc_score(labels.cpu().numpy(), score_probs.cpu().numpy(), multi_class='ovr')
        f1 = f1_score(labels.cpu().numpy(), scores.cpu().numpy(), average='weighted')
        accuracy = accuracy_score(labels.cpu().numpy(), scores.cpu().numpy())
        recall = recall_score(labels.cpu().numpy(), scores.cpu().numpy(), average='weighted')

        return {
            'roc_auc': roc_auc,
            'f1-score': f1,
            'recall': recall,
            'accuracy': accuracy
        }
    
    def plot_metrics(self):
        results = pd.DataFrame(self.results)
        df = pd.DataFrame(results, index=range(1, len(results)+1))
        df['epoch'] = df.index
        df.loc[df['phase'] == 'test'].plot(x='epoch', y=['roc_auc', 'f1-score', 'accuracy', 'recall', 'loss'], kind='line')


    def run_epochs(self, num_epochs):
        for _ in tqdm(range(num_epochs), desc="Running Epochs"):
            self.train()
            self.test()
        
