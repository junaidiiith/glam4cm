import torch.nn.functional as F
from torch_geometric.nn import (
    global_mean_pool,
    global_max_pool,
    global_add_pool,
)
from torch import nn
import torch
import json
import os
import torch_geometric
from settings import device
from torch_geometric.nn import APPNP

pooling_methods = {
    'mean': global_mean_pool,
    'sum': global_add_pool,
    'max': global_max_pool,
}

supported_conv_models = {
    'GCNConv': False, ## True or False if the model requires num_heads
    'GraphConv': False,
    'GATConv': True,
    'SAGEConv': False,
    'GINConv': False,
    'GatedGraphConv': False,
    'GATv2Conv': True,
}


class GNNConv(torch.nn.Module):
    """
        A general GNN model created using the PyTorch Geometric library
        model_name: the name of the GNN model
        input_dim: the input dimension
        hidden_dim: the hidden dimension
        out_dim: the output dimension

        num_layers: the number of GNN layers
        num_heads: the number of heads in the GNN layer
        residual: whether to use residual connections
        l_norm: whether to use layer normalization
        dropout: the dropout probability
    
    """
    def __init__(
            self, 
            model_name, 
            input_dim, 
            hidden_dim,  
            num_layers, 
            output_dim=None,
            num_heads=None, 
            residual=False, 
            l_norm=False, 
            dropout=0.1,
        ):
        super(GNNConv, self).__init__()

        assert model_name in supported_conv_models, f"Model {model_name} not supported"
        should_have_heads = supported_conv_models[model_name]

        if should_have_heads:
            assert num_heads is not None, f"Model {model_name} requires num_heads"
        else:
            assert num_heads is None, f"Model {model_name} does not require num_heads"
        
        self.input_dim = input_dim
        self.embed_dim = hidden_dim
        self.output_dim = output_dim if output_dim else hidden_dim
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.residual = residual
        self.l_norm = l_norm
        self.dropout = dropout
        

        gnn_model = getattr(torch_geometric.nn, model_name)
        self.conv_layers = nn.ModuleList()
        if model_name == 'GINConv':
            input_layer = gnn_model(
                nn.Sequential(
                    nn.Linear(input_dim, hidden_dim), 
                    nn.ReLU()), 
                    train_eps=True
                )
        elif num_heads is None:
            input_layer = gnn_model(
                input_dim, 
                hidden_dim, 
                aggr='SumAggregation',
                
            )
        else:
            input_layer = gnn_model(
                input_dim, 
                hidden_dim, 
                heads=num_heads, 
                aggr='SumAggregation',
                
            )
        self.conv_layers.append(input_layer)

        for _ in range(num_layers - 2):
            if model_name == 'GINConv':
                self.conv_layers.append(
                    gnn_model(
                        nn.Sequential(
                            nn.Linear(hidden_dim, hidden_dim), 
                            nn.ReLU()), 
                            train_eps=True
                        )
                    )
            elif num_heads is None:
                self.conv_layers.append(
                    gnn_model(
                        hidden_dim, 
                        hidden_dim, 
                        aggr='SumAggregation',
                        
                    )
                )
            else:
                self.conv_layers.append(
                    gnn_model(
                        num_heads*hidden_dim, 
                        hidden_dim, 
                        heads=num_heads, 
                        aggr='SumAggregation',
                        
                    )
                )

        output_dim = output_dim if output_dim else hidden_dim
        if model_name == 'GINConv':
            self.conv_layers.append(
                gnn_model(
                    nn.Sequential(
                        nn.Linear(hidden_dim, output_dim), 
                        nn.ReLU()), 
                        train_eps=True
                    )
                )
        else:
            self.conv_layers.append(
                gnn_model(
                    hidden_dim if num_heads is None else num_heads*hidden_dim, 
                    output_dim, 
                    aggr='SumAggregation',
                    
                )
            )
            
        self.activation = nn.ReLU()
        self.layer_norm = nn.LayerNorm(
            hidden_dim if num_heads is None else num_heads*hidden_dim
        ) if l_norm else None
        self.residual = residual
        self.dropout = nn.Dropout(dropout)


    def forward(self, in_feat, edge_index):
        h = in_feat
        h = self.conv_layers[0](h, edge_index)
        h = self.activation(h)
        if self.layer_norm is not None:
            h = self.layer_norm(h)
        h = self.dropout(h)

        for conv in self.conv_layers[1:-1]:
            h = conv(h, edge_index) if not self.residual else conv(h, edge_index) + h
            h = self.activation(h)
            if self.layer_norm is not None:
                h = self.layer_norm(h)
            h = self.dropout(h)
        
        h = self.conv_layers[-1](h, edge_index)
        return h
  

    def save_pretrained(self, path):
        os.makedirs(path, exist_ok=True)
        state_dict_path = f'{path}/gnn_state_dict.pt'
        config_path = f'{path}/gnn_config.json'
        with open(config_path, 'w') as f:
            json.dump(
                {
                    "model_name":"SAGEConv", 
                    "input_dim": self.input_dim,
                    "embed_dim": self.embed_dim,
                    "out_dim": self.output_dim,
                    "num_layers": self.num_layers,
                    "num_heads": self.num_heads,
                    "residual": self.residual,
                    "l_norm": self.l_norm,
                    "dropout": self.dropout.p,
                }, f)
    
        torch.save(self.state_dict(), state_dict_path)
        print(f'Saved GNN model at {path}')


    @staticmethod
    def from_pretrained(state_dict_dir):
        state_dict = torch.load(f"{state_dict_dir}/gnn_state_dict.pt", map_location=device)
        gnn_model_config = json.load(open(f"{state_dict_dir}/gnn_config.json", 'r'))
        gnn_model = GNNConv(
            model_name=gnn_model_config['model_name'],
            input_dim=gnn_model_config['input_dim'],
            hidden_dim=gnn_model_config['embed_dim'],
            out_dim=gnn_model_config['out_dim'],
            num_heads=gnn_model_config['num_heads'],
            num_layers=gnn_model_config['num_layers'],
            residual=gnn_model_config['residual'],
            dropout=gnn_model_config['dropout'],
        )
        gnn_model.load_state_dict(state_dict)
        return gnn_model


class GNNClassifier(nn.Module):
    def __init__(
            self,
            gnn_conv_model, 
            input_dim, 
            hidden_dim, 
            output_dim, 
            num_layers, 
            num_heads=None,
            dropout=0.1,
            residual = False,
            pool = 'mean',
            use_appnp = False,
            **kwargs
        ):
        super(GNNClassifier, self).__init__()
        self.conv = GNNConv(
            model_name=gnn_conv_model,
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            num_heads=num_heads,
            dropout=dropout,
            residual=residual,
        )

        self.fc = nn.Linear(hidden_dim, output_dim)
        if pool not in pooling_methods:
            raise ValueError(f"Pooling method {pool} not supported")
        self.pool = pooling_methods[pool]

        if use_appnp:
            assert 'K' in kwargs, "K not provided for APPNP"
            assert 'alpha' in kwargs, "Alpha not provided for APPNP"
            self.propagate = APPNP(kwargs['K'], kwargs['alpha'])


    def forward(self, x, edge_index, batch):
        h = self.conv(x, edge_index)
        h = self.pool(h, batch)
        h = self.fc(h)
        return F.log_softmax(h, dim=-1)
