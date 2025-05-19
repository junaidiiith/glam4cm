from sklearn.metrics import roc_auc_score, average_precision_score
from torch_geometric.loader import DataLoader
from torch_geometric.data import Data
from torch_geometric.nn import GATConv, VGAE
import torch.nn.functional as F
import torch
from typing import List

from glam4cm.models.gnn_layers import (
    GNNConv, 
    EdgeClassifer
)

from tqdm.auto import tqdm
from glam4cm.settings import device


class GATVGAEEncoder(torch.nn.Module):
    def __init__(self, in_channels, hid_channels, out_channels, heads=(4,2), dropout=0.3):
        super().__init__()
        self.conv1 = GATConv(in_channels, hid_channels, heads=heads[0], dropout=dropout)
        # mu and log_std each map to latent dim
        self.conv_mu     = GATConv(hid_channels * heads[0], out_channels, heads=heads[1], concat=False)
        self.conv_logstd = GATConv(hid_channels * heads[0], out_channels, heads=heads[1], concat=False)

    def forward(self, x, edge_index):
        x = F.elu(self.conv1(x, edge_index))
        return self.conv_mu(x, edge_index), self.conv_logstd(x, edge_index)



class GNNLinkPredictionTrainerV2:
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
        
        self.num_epochs = num_epochs
        self.lr = lr
        in_dim = dataset[0].data.x.shape[1]
        hid_dim = 64
        out_dim = 32
        self.encoder = GATVGAEEncoder(in_dim, hid_dim, out_dim).to(device)
        self.model   = VGAE(self.encoder).to(device)
        self.opt     = torch.optim.Adam(model.parameters(), lr=lr)

        self.dataloader = DataLoader(
            dataset, batch_size=batch_size, shuffle=True
        )
        self.results = list()

        print("GNN Trainer initialized.")



    def train(self):
        self.model.train()
        total_loss = 0.0

        for data in self.dataloader:
            data = data.to(device)
            # Encode over the **train positive** graph only:
            z = self.model.encode(data.x, data.train_pos_edge_label_index)

            # recon_loss only on positives:
            loss = self.model.recon_loss(z, data.train_pos_edge_label_index)
            # KL regularizer:
            loss += (1. / data.num_nodes) * self.model.kl_loss()

            self.opt.zero_grad()
            loss.backward()
            self.opt.step()

            total_loss += loss.item()

        return total_loss / len(self.dataloader)


    @torch.no_grad()
    def test(self):
        self.model.eval()
        all_auc, all_ap = [], []

        for data in self.dataloader:
            data = data.to(device)
            z = self.model.encode(data.x, data.train_pos_edge_label_index)

            # positive edges from your test split
            pos_idx = data.test_pos_edge_label_index
            # generate equal‐size negative sample
            neg_idx = data.test_neg_edge_label_index

            pos_scores = self.model.decoder(z, pos_idx).sigmoid()
            neg_scores = self.model.decoder(z, neg_idx).sigmoid()

            y_true  = torch.cat([torch.ones(pos_scores.size(0)),
                                torch.zeros(neg_scores.size(0))]).cpu()
            y_score = torch.cat([pos_scores, neg_scores]).cpu()

            all_auc.append( roc_auc_score(y_true, y_score) )
            all_ap.append(  average_precision_score(y_true, y_score) )

        return {
            'AUC': sum(all_auc) / len(all_auc),
            'AP':  sum(all_ap)  / len(all_ap),
        }
        

    def compute_loss(self, pos_score, neg_score):
        pos_label = torch.ones(pos_score.size(0), dtype=torch.long).to(device)
        neg_label = torch.zeros(neg_score.size(0), dtype=torch.long).to(device)

        scores = torch.cat([pos_score, neg_score], dim=0)
        labels = torch.cat([pos_label, neg_label], dim=0)

        loss = self.criterion(scores, labels)
        return loss
    
    
    def run(self):
        all_metrics = list()
        for epoch in tqdm(range(self.num_epochs), desc="Running Epochs"):
            self.train()
            test_metrics = self.test()
            all_metrics.append(test_metrics)
            print(f"Epoch {epoch+1}/{self.num_epochs} | AUC: {test_metrics['AUC']:.4f} | AP: {test_metrics['AP']:.4f}")

        print("Training complete.")
        best_metrics = sorted(all_metrics, key=lambda x: x['AUC'], reverse=True)[0]
        
        s2t = lambda x: x.replace("_", " ").title()
        print(f"Best: {' | '.join([f'{s2t(k)}: {v:.4f}' for k, v in best_metrics.items()])}")
    