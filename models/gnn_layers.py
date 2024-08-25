import torch
from torch.nn import functional as F
from torch_geometric.nn import aggr
from torch_geometric.nn import (
    global_add_pool,
    global_max_pool,
    global_mean_pool,
)
import torch_geometric
import torch.nn as nn


aggregation_methods = {
    'mean': aggr.MeanAggregation(),
    'sum': aggr.SumAggregation(),
    'max': aggr.MaxAggregation(),
    'mul': aggr.MulAggregation(),
}

supported_conv_models = {
    'GCNConv': False, ## True or False if the model requires num_heads
    'GraphConv': False,
    'GATConv': True,
    'SAGEConv': False,
    'GINConv': False,
    'GATv2Conv': True,
}

global_pooling_methods = {
    'sum': global_add_pool,
    'mean': global_mean_pool,
    'max': global_max_pool,
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
            out_dim, 
            num_layers, 
            num_heads=None, 
            residual=False, 
            l_norm=False, 
            dropout=0.1,
            aggregation='sum',
            edge_dim=None
        ):
        super(GNNConv, self).__init__()

        assert model_name in supported_conv_models, f"Model {model_name} not supported. Choose from {supported_conv_models.keys()}"
        heads_supported = supported_conv_models[model_name]
        if heads_supported and num_heads is None:
            raise ValueError(f"Model {model_name} requires num_heads to be set to an integer")
        
        if not heads_supported and num_heads is not None:
            num_heads = None

        assert aggregation in aggregation_methods, f"Aggregation method {aggregation} not supported. Choose from {aggregation_methods.keys()}"
        aggregation = aggregation_methods[aggregation]

        self.input_dim = input_dim
        self.embed_dim = hidden_dim
        self.out_dim = out_dim
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.residual = residual
        self.l_norm = l_norm
        self.dropout = dropout
        self.aggregation = aggregation
        self.edge_dim = edge_dim
        

        gnn_model = getattr(torch_geometric.nn, model_name)
        self.conv_layers = nn.ModuleList()
        if num_heads is None:
            input_layer = gnn_model(input_dim, hidden_dim, aggr=aggregation)
        else:
            input_layer = gnn_model(
                input_dim, 
                hidden_dim, 
                heads=num_heads, 
                aggr=aggregation,
                edge_dim=edge_dim
            )
        self.conv_layers.append(input_layer)

        for _ in range(num_layers - 2):
            if num_heads is None:
                conv = gnn_model(hidden_dim, hidden_dim, aggr=aggregation)
                self.conv_layers.append(conv)
            else:
                conv = gnn_model(
                    num_heads*hidden_dim, 
                    hidden_dim, 
                    heads=num_heads, 
                    aggr=aggregation,
                    edge_dim=edge_dim
                )
                self.conv_layers.append(conv)

        if num_heads is None:
            self.conv_layers.append(gnn_model(hidden_dim, out_dim, aggr=aggregation))
        else:
            self.conv_layers.append(gnn_model(
                num_heads*hidden_dim, 
                out_dim, 
                heads=num_heads, 
                aggr=aggregation,
                edge_dim=edge_dim
            ))
            
        self.activation = nn.ReLU()
        self.layer_norm = nn.LayerNorm(hidden_dim if num_heads is None else num_heads*hidden_dim) if l_norm else None
        self.residual = residual
        self.dropout = nn.Dropout(dropout)


    def forward(self, in_feat, edge_index, edge_attr=None):
        h = in_feat
        h = self.conv_layers[0](h, edge_index, edge_attr) if edge_attr is not None else self.conv_layers[0](h, edge_index)
        h = self.activation(h)
        if self.layer_norm is not None:
            h = self.layer_norm(h)
        h = self.dropout(h)

        for conv in self.conv_layers[1:-1]:
            nh = conv(h, edge_index, edge_attr) if edge_attr is not None else conv(h, edge_index)
            h = nh if not self.residual else nh + h
            h = self.activation(h)
            if self.layer_norm is not None:
                h = self.layer_norm(h)
            h = self.dropout(h)
        
        h = self.conv_layers[-1](h, edge_index) if edge_attr is not None else self.conv_layers[-1](h, edge_index)
        return h
  

class EdgeClassifer(nn.Module):

    """
    An MLP predictor for link prediction

    h_feats: the input dimension
    num_classes: the number of classes
    num_layers: the number of layers in the MLP

    This class concatenates the node embeddings of the two nodes in the edge
    The concatenated embeddings are then passed through an MLP
    """

    def __init__(
            self, 
            input_dim,
            hidden_dim, 
            num_classes,
            num_layers=2, 
            dropout=0.3,
            edge_dim=None,
            bias=True
        ):
        super().__init__()
        self.layers = nn.ModuleList()
        self.input_dim = input_dim
        self.embed_dim = hidden_dim
        self.num_layers = num_layers
        self.num_classes = num_classes

        in_feats = input_dim * 2
        if edge_dim is not None:
            in_feats += edge_dim
            
        for _ in range(num_layers - 1):
            self.layers.append(nn.Linear(in_feats, hidden_dim, bias=bias))
            self.layers.append(nn.ReLU())
            self.layers.append(nn.Dropout(dropout))
            in_feats = hidden_dim
        
        self.layers.append(nn.Linear(hidden_dim, num_classes, bias=bias))


    def forward(self, x, edge_index, edge_attr=None):
        h = torch.cat([x[edge_index[0]], x[edge_index[1]]], dim=-1)
        if edge_attr is not None:
            h = torch.cat([h, edge_attr], dim=-1)
            
        for layer in self.layers:
            h = layer(h)
        
        return F.softmax(h, dim=-1)
    

class NodeClassifier(nn.Module):

    """
    An MLP predictor for link prediction

    h_feats: the input dimension
    num_classes: the number of classes
    num_layers: the number of layers in the MLP

    This class concatenates the node embeddings of the two nodes in the edge
    The concatenated embeddings are then passed through an MLP
    """

    def __init__(
            self, 
            input_dim,
            hidden_dim, 
            num_classes,
            num_layers=2, 
            dropout=0.3,
            bias=True
        ):
        super().__init__()
        self.layers = nn.ModuleList()
        self.embed_dim = hidden_dim
        self.num_layers = num_layers
        self.num_classes = num_classes

        for _ in range(num_layers - 1):
            self.layers.append(nn.Linear(input_dim, hidden_dim, bias=bias))
            self.layers.append(nn.ReLU())
            self.layers.append(nn.Dropout(dropout))
            input_dim = hidden_dim
        
        self.layers.append(nn.Linear(hidden_dim, num_classes, bias=bias))


    def forward(self, x):
        h = x
        for layer in self.layers:
            h = layer(h)
        
        return F.log_softmax(h, dim=-1)
    

class GraphClassifer(nn.Module):

    """
    An MLP predictor for link prediction

    h_feats: the input dimension
    num_classes: the number of classes
    num_layers: the number of layers in the MLP

    This class concatenates the node embeddings of the two nodes in the edge
    The concatenated embeddings are then passed through an MLP
    """

    def __init__(
            self, 
            input_dim, 
            num_classes,
            global_pool='mean',
            bias=False
        ):
        super().__init__()
        self.layers = nn.ModuleList()
        self.input_dim = input_dim
        self.num_classes = num_classes

        self.layers.append(nn.Linear(input_dim, num_classes, bias=bias))
        self.global_pool = global_pooling_methods[global_pool]

    def forward(self, x, batch):
        h = self.global_pool(x, batch)
        for layer in self.layers:
            h = layer(h)
        
        return h.log_softmax(dim=-1)
