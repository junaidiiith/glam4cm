from sklearn.metrics import (
    balanced_accuracy_score,
    f1_score, 
    recall_score, 
    accuracy_score
)
import torch
from collections import defaultdict
from torch_geometric.loader import DataLoader
from random import shuffle

from data_loading.graph_dataset import GraphEdgeDataset
from models.gnn_layers import (
    GNNConv,
    GraphClassifer
)
from utils import get_device, randomize_features
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
            dataset: GraphEdgeDataset,
            tr=0.2,
            lr=1e-4,
            num_epochs=100,
            batch_size=32,
            randomize_ne = False,
            ne_features=768
        ) -> None:

        super().__init__(
            model=model,
            predictor=predictor,
            cls_label='type',
            lr=lr,
            num_epochs=num_epochs
        )

        tr = 1 - tr
        self.dataloaders = dict()
        dataset = [g.data for g in dataset]
        shuffle(dataset)
        if randomize_ne:
            dataset = randomize_features(dataset, ne_features)

        train_dataset = dataset[:int(tr * len(dataset))]
        test_dataset = dataset[int(tr * len(dataset)):]
        self.dataloaders['train'] = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        self.dataloaders['test'] = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

        print("GNN Trainer initialized.")


    def train(self):
        self.model.train()
        self.predictor.train()

        epoch_loss = 0
        epoch_metrics = defaultdict(float)
        preds, labels = list(), list()
        # for i, data in tqdm(enumerate(self.dataloader), desc=f"Training batches", total=len(self.dataloader)):
        for data in self.dataloaders['train']:
            self.optimizer.zero_grad()
            self.model.train()
            self.predictor.train()
            
            h = self.model(data.x.to(device), data.edge_index.to(device))
            g_pred = self.predictor(h, data.batch.to(device))

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
        self.model.eval()
        self.classifier.eval()
        with torch.no_grad():
            epoch_loss = 0
            preds, labels = list(), list()
            for data in self.dataloaders['test']:
                h = self.model(data.x.to(device), data.edge_index.to(device))
                g_pred = self.predictor(h, data.batch.to(device))

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