import torch
import torch.nn.functional as F
from torch_geometric.nn import SAGEConv, global_mean_pool
from torch_geometric.loader import DataLoader
from torch import nn
from settings import device


class GraphSAGEBlock(nn.Module):
    def __init__(self, in_dim, out_dim, dropout):
        super(GraphSAGEBlock, self).__init__()
        self.conv = SAGEConv(in_dim, out_dim, aggr='sum')
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x, edge_index):
        x = self.conv(x, edge_index)
        x = self.relu(x)
        x = self.dropout(x)
        return x


class GraphSAGE(nn.Module):
    def __init__(
            self, 
            input_dim, 
            hidden_dim, 
            output_dim, 
            num_layers, 
            dropout=0.1
        ):
        super(GraphSAGE, self).__init__()
        self.gnn = nn.ModuleList([
            GraphSAGEBlock(
                input_dim if i == 0 else hidden_dim, 
                hidden_dim, 
                dropout
            ) for i in range(num_layers)
        ])
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x, edge_index, batch):
        for gnn_layer in self.gnn:
            x = gnn_layer(x, edge_index)
        x = global_mean_pool(x, batch)
        x = self.fc(x)
        return F.log_softmax(x, dim=1)
