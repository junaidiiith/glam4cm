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

from torch_geometric.nn import (
    GATv2Conv,
    SAGEConv
)

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


from settings import device
from torch_geometric.nn import (
    global_mean_pool,
    global_max_pool,
    global_add_pool,
)
from torch import nn

from torch_geometric.nn import (
    GATv2Conv,
    SAGEConv
)

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


class FeedForward(nn.Module):
    """
    Create a feed forward neural network with ReLU activation
    Use n hidden layers with hidden_dim units each
    Use output_dim units in the output layer
    Use Sequential to create the feed forward network
    """

    def __init__(
            self, 
            input_dim, 
            hidden_dim=None, 
            output_dim=None,
            use_bias=False, 
            num_layers=3, 
            dropout=0.1,
            final_activation=None
        ):
        super(FeedForward, self).__init__()

        hidden_dim = hidden_dim if hidden_dim else input_dim
        output_dim = output_dim if output_dim else hidden_dim

        current_power = 2**num_layers

        layers = nn.ModuleList()
        layers.append(nn.Linear(input_dim, hidden_dim*current_power, bias=use_bias))
        layers.append(nn.ReLU())
        layers.append(nn.Dropout(dropout))
        for _ in range(num_layers - 2):
            layers.append(nn.Linear(hidden_dim*current_power, hidden_dim*(current_power//2), bias=use_bias))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            current_power = current_power // 2

        layers.append(nn.Linear(hidden_dim*current_power, output_dim, bias=use_bias))

        if final_activation == 'softmax':
            layers.append(nn.Softmax(dim=-1))
        elif final_activation == 'sigmoid':
            layers.append(nn.Sigmoid())
        
        self.ff = nn.Sequential(*layers)

    def forward(self, x):
        return self.ff(x)


class GATv2(nn.Module):
    def __init__(
        self,
        input_dim,
        hidden_dim=None,
        output_dim=None,
        num_layers=2,
        num_heads=1,
        dropout=0.1,
        edge_dim=None,
        concat=True,
        residual=False,
    ):
        super(GATv2, self).__init__()
        self.input_dim = input_dim
        self.embed_dim = hidden_dim
        self.output_dim = output_dim if output_dim else hidden_dim
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.dropout = dropout
        self.edge_dim = edge_dim
        self.concat = concat
        self.residual = residual


        self.conv_layers = nn.ModuleList()
        input_layer = GATv2Conv(
            input_dim,
            hidden_dim,
            heads=num_heads,
            concat=concat,
            dropout=dropout,
            edge_dim=edge_dim,
        )
        self.conv_layers.append(input_layer)

        for _ in range(num_layers - 2):
            self.conv_layers.append(
                GATv2Conv(
                    hidden_dim*num_heads,
                    hidden_dim,
                    heads=num_heads,
                    concat=concat,
                    dropout=dropout,
                    edge_dim=edge_dim,
                )
            )

        output_dim = output_dim if output_dim else hidden_dim
        self.conv_layers.append(
            GATv2Conv(
                hidden_dim*num_heads,
                output_dim,
                heads=num_heads,
                concat=concat,
                dropout=dropout,
                edge_dim=edge_dim,
            )
        )
        self.activation = nn.ReLU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, edge_index, edge_attr):
        h = x
        h = self.conv_layers[0](h, edge_index, edge_attr)
        h = self.activation(h)
        h = self.dropout(h)

        for conv in self.conv_layers[1:-1]:
            h = conv(h, edge_index, edge_attr) + h if self.residual else conv(h, edge_index, edge_attr)
            h = self.activation(h)
            h = self.dropout(h)
        
        h = self.conv_layers[-1](h, edge_index, edge_attr)
        return h



class SAGE(nn.Module):
    def __init__(
        self,
        input_dim,
        hidden_dim=None,
        output_dim=None,
        num_layers=2,
        dropout=0.1,
        residual=False,
    ):
        super(SAGE, self).__init__()
        self.input_dim = input_dim
        self.embed_dim = hidden_dim
        self.output_dim = output_dim if output_dim else hidden_dim
        self.num_layers = num_layers
        self.dropout = dropout
        self.residual = residual

        self.conv_layers = nn.ModuleList()
        input_layer = SAGEConv(
            input_dim,
            hidden_dim,
            aggr='mean',
        )
        self.conv_layers.append(input_layer)

        for _ in range(num_layers - 2):
            self.conv_layers.append(
                SAGEConv(
                    hidden_dim,
                    hidden_dim,
                    aggr='mean',
                )
            )

        output_dim = output_dim if output_dim else hidden_dim
        self.conv_layers.append(
            SAGEConv(
                hidden_dim,
                output_dim,
                aggr='mean',
            )
        )
        self.activation = nn.ReLU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, edge_index):
        h = x
        h = self.conv_layers[0](h, edge_index)
        h = self.activation(h)
        h = self.dropout(h)

        for conv in self.conv_layers[1:-1]:
            h = conv(h, edge_index) + h if self.residual else conv(h, edge_index)
            h = self.activation(h)
            h = self.dropout(h)
        
        h = self.conv_layers[-1](h, edge_index)
        return h


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
            edge_dim=None,
            concat=False
        ):
        super(GNNConv, self).__init__()
        self.model_name = model_name

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

        self.edge_attr = edge_dim
        self.edge_dim = edge_dim
        self.concat = concat
        

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
                edge_dim=edge_dim,
                concat=concat,
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
        elif num_heads is None:
            self.conv_layers.append(
                gnn_model(
                    hidden_dim if num_heads is None else num_heads*hidden_dim, 
                    output_dim, 
                    aggr='SumAggregation',
                    
                )
            )
        else:
            self.conv_layers.append(
                gnn_model(
                    num_heads*hidden_dim, 
                    output_dim, 
                    heads=num_heads, 
                    aggr='SumAggregation',
                    
                )
            )
            
        self.activation = nn.ReLU()
        self.layer_norm = nn.LayerNorm(
            hidden_dim if num_heads is None else num_heads*hidden_dim
        ) if l_norm else None
        self.residual = residual
        self.dropout = nn.Dropout(dropout)


    def forward(self, in_feat, edge_index, edge_attr=None):
        h = in_feat
        h = self.conv_layers[0](h, edge_index, edge_attr)
        h = self.activation(h)
        if self.layer_norm is not None:
            h = self.layer_norm(h)
        h = self.dropout(h)

        for conv in self.conv_layers[1:-1]:
            h = conv(h, edge_index, edge_attr) if not self.residual else conv(h, edge_index, edge_attr) + h
            h = self.activation(h)
            if self.layer_norm is not None:
                h = self.layer_norm(h)
            h = self.dropout(h)
        
        h = self.conv_layers[-1](h, edge_index, edge_attr)
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


class GNNLinkPredictor(nn.Module):
    def __init__(self, gnn_conv, lp_head, ec_head):
        super(GNNLinkPredictor, self).__init__()
        self.gnn_conv = gnn_conv
        self.lp_head = lp_head
        self.ec_head = ec_head

        self.lp_criterion = torch.nn.BCEWithLogitsLoss()
        self.ec_criterion = torch.nn.CrossEntropyLoss()


    def forward(self, x, edge_index, edge_attr):
        h = self.gnn_conv(x, edge_index, edge_attr)
        return h
    

    def get_link_predictions(self, h, pos_edge_index, neg_edge_index):
        row, col = pos_edge_index
        pos_edge_features = torch.cat([h[row], h[col]], dim=1)  # [num_edges, hidden_channels*2]
        pos_link_pred = torch.sigmoid(self.lp_head(pos_edge_features)).squeeze()  # [num_edges]

        # Negative sampling
        row, col = neg_edge_index
        neg_edge_features = torch.cat([h[row], h[col]], dim=1)
        neg_link_pred = torch.sigmoid(self.lp_head(neg_edge_features)).squeeze()  # [num_edges]

        return pos_link_pred, neg_link_pred
    

    def link_predictor_loss(self, h, pos_edge_index, neg_edge_index):

        pos_link_pred, neg_link_pred = self.get_link_predictions(
            h, pos_edge_index, neg_edge_index
        )

        pos_labels = torch.ones_like(pos_link_pred)
        neg_labels = torch.zeros_like(neg_link_pred)

        pos_loss = self.lp_criterion(pos_link_pred, pos_labels)
        neg_loss = self.lp_criterion(neg_link_pred, neg_labels)

        return pos_loss + neg_loss


    def get_edge_classifier_predictions(self, h, edge_index):
        row, col = edge_index
        ec_features = torch.cat([h[row], h[col]], dim=1)
        ec_pred = self.ec_head(ec_features)
        return ec_pred


    def edge_classifier_loss(self, h, edge_index, edge_classes):
        row, col = edge_index
        ec_features = torch.cat([h[row], h[col]], dim=1)
        ec_pred = self.ec_head(ec_features)
        return self.ec_criterion(ec_pred, edge_classes)
    

    def save_pretrained(self, path):
        os.makedirs(path, exist_ok=True)
        state_dict_path = f'{path}/link_predictor_state_dict.pt'
        config_path = f'{path}/link_predictor_config.json'
        with open(config_path, 'w') as f:
            json.dump(
                {
                    "gnn_conv_model": self.gnn_conv.model_name,
                    "lp_head": self.lp_head.state_dict(),
                    "ec_head": self.ec_head.state_dict(),
                }, f)
    
        torch.save(self.state_dict(), state_dict_path)
        print(f'Saved Link Predictor model at {path}')
    

    @staticmethod
    def from_pretrained(state_dict_dir):
        state_dict = torch.load(f"{state_dict_dir}/link_predictor_state_dict.pt", map_location=device)
        link_predictor_config = json.load(open(f"{state_dict_dir}/link_predictor_config.json", 'r'))
        gnn_conv = GNNConv.from_pretrained(state_dict_dir)
        lp_head = link_predictor_config['lp_head']
        ec_head = link_predictor_config['ec_head']
        link_predictor = GNNLinkPredictor(gnn_conv, lp_head, ec_head)
        link_predictor.load_state_dict(state_dict)
        return link_predictor
    


if __name__ == '__main__':
    input_dim = 768
    hidden_dim = 128
    output_dim = 128
    num_layers = 3
    num_heads = 4
    edge_dim = 768
    residual = True


    gat = GATv2(
        input_dim=input_dim,
        hidden_dim=hidden_dim,
        output_dim=output_dim,
        num_layers=num_layers,
        num_heads=num_heads,
        edge_dim=edge_dim,
        residual=residual,
        dropout=0.2,
    )

    lp_head = FeedForward(
        input_dim=(output_dim*num_heads if num_heads > 1 else output_dim) * 2,
        hidden_dim=hidden_dim,
        output_dim=1,
        num_layers=3,
    )

    ec_head = FeedForward(
        input_dim=(output_dim*num_heads if num_heads > 1 else output_dim) * 2,
        hidden_dim=hidden_dim,
        output_dim=3,
        num_layers=3,
        final_activation='softmax',
    )

    lp_head.to(device), ec_head.to(device)