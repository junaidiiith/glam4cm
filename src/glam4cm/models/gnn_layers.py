"""
Graph Neural Network Layers for GLAM4CM

This module provides a comprehensive collection of Graph Neural Network (GNN) layers
and architectures specifically designed for conceptual modeling tasks. The module
supports various GNN variants including GCN, GraphSAGE, GAT, GIN, and GraphConv.

Key Features:
- Multiple GNN layer types with consistent interfaces
- Configurable aggregation methods (mean, sum, max, mul)
- Support for edge features and multi-head attention
- Global pooling operations for graph-level tasks
- Residual connections and layer normalization
- Comprehensive model validation and error handling

Supported Models:
- GCNConv: Graph Convolutional Networks
- GraphConv: Generic Graph Convolution
- GATConv: Graph Attention Networks
- SAGEConv: GraphSAGE with sampling
- GINConv: Graph Isomorphism Networks
- GATv2Conv: Improved Graph Attention Networks

Author: Syed Juned Ali
Email: syed.juned.ali@tuwien.ac.at
"""

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


# =============================================================================
# AGGREGATION METHODS
# =============================================================================

# Dictionary mapping aggregation method names to PyTorch Geometric aggregators
aggregation_methods = {
    "mean": aggr.MeanAggregation(),  # Average aggregation
    "sum": aggr.SumAggregation(),  # Sum aggregation
    "max": aggr.MaxAggregation(),  # Maximum aggregation
    "mul": aggr.MulAggregation(),  # Element-wise multiplication
}

# =============================================================================
# SUPPORTED CONVOLUTION MODELS
# =============================================================================

# Dictionary indicating which models require num_heads parameter
# True: Model supports multi-head attention (e.g., GAT)
# False: Model doesn't use attention heads
supported_conv_models = {
    "GCNConv": False,  # Graph Convolutional Networks
    "GraphConv": False,  # Generic Graph Convolution
    "GATConv": True,  # Graph Attention Networks
    "SAGEConv": False,  # GraphSAGE
    "GINConv": False,  # Graph Isomorphism Networks
    "GATv2Conv": True,  # Improved Graph Attention Networks
}

# =============================================================================
# GLOBAL POOLING METHODS
# =============================================================================

# Dictionary mapping pooling method names to PyTorch Geometric functions
global_pooling_methods = {
    "sum": global_add_pool,  # Sum all node features
    "mean": global_mean_pool,  # Average all node features
    "max": global_max_pool,  # Maximum across all node features
}


class GNNConv(torch.nn.Module):
    """
    General Graph Neural Network model using PyTorch Geometric.

    This class provides a unified interface for various GNN architectures,
    allowing easy switching between different convolution types while
    maintaining consistent hyperparameter interfaces.

    Architecture:
        - Configurable number of GNN layers
        - Support for different convolution types
        - Configurable aggregation methods
        - Optional residual connections
        - Optional layer normalization
        - Dropout for regularization

    Supported Convolution Types:
        - GCNConv: Standard graph convolution
        - GATConv: Graph attention with multi-head support
        - SAGEConv: GraphSAGE with neighbor sampling
        - GINConv: Graph isomorphism networks
        - GraphConv: Generic graph convolution
        - GATv2Conv: Improved graph attention
    """

    def __init__(
        self,
        model_name,
        input_dim,
        hidden_dim,
        out_dim=None,
        num_layers=2,
        num_heads=None,
        residual=False,
        l_norm=False,
        dropout=0.1,
        aggregation="mean",
        edge_dim=None,
    ):
        """
        Initialize the GNN model.

        Args:
            model_name: Name of the GNN convolution layer to use
            input_dim: Input feature dimension
            hidden_dim: Hidden layer dimension
            out_dim: Output dimension (defaults to hidden_dim if None)
            num_layers: Number of GNN layers
            num_heads: Number of attention heads (required for attention-based models)
            residual: Whether to use residual connections
            l_norm: Whether to use layer normalization
            dropout: Dropout probability for regularization
            aggregation: Aggregation method ('mean', 'sum', 'max', 'mul')
            edge_dim: Edge feature dimension (if using edge features)

        Raises:
            ValueError: If model_name is not supported or num_heads is missing for attention models
            AssertionError: If aggregation method is not supported
        """
        super(GNNConv, self).__init__()

        # Validate model name and requirements
        assert model_name in supported_conv_models, (
            f"Model {model_name} not supported. Choose from {supported_conv_models.keys()}"
        )
        heads_supported = supported_conv_models[model_name]

        if heads_supported and num_heads is None:
            raise ValueError(
                f"Model {model_name} requires num_heads to be set to an integer"
            )

        if not heads_supported and num_heads is not None:
            num_heads = None

        # Validate aggregation method
        assert aggregation in aggregation_methods, (
            f"Aggregation method {aggregation} not supported. Choose from {aggregation_methods.keys()}"
        )
        aggregation = aggregation_methods[aggregation]

        # Store configuration parameters
        self.input_dim = input_dim
        self.embed_dim = hidden_dim
        self.out_dim = out_dim if out_dim is not None else hidden_dim
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.aggregation = aggregation
        self.edge_dim = edge_dim
        self.residual = residual
        self.l_norm = l_norm
        self.dropout = dropout

        # Get the GNN model class from PyTorch Geometric
        gnn_model = getattr(torch_geometric.nn, model_name)
        self.conv_layers = nn.ModuleList()

        # Build GNN layers
        for i in range(num_layers):
            # Determine input and output dimensions for each layer
            if num_heads is None:
                # For non-attention models
                conv = gnn_model(
                    input_dim,
                    hidden_dim if i != num_layers - 1 else self.out_dim,
                    aggr=aggregation,
                )
            else:
                # For attention-based models (GAT, GATv2)
                conv = gnn_model(
                    input_dim if i == 0 else num_heads * input_dim,
                    hidden_dim if i != num_layers - 1 else self.out_dim,
                    heads=num_heads,
                    aggr=aggregation,
                )

            self.conv_layers.append(conv)

        # Add layer normalization if requested
        if l_norm:
            self.layer_norms = nn.ModuleList(
                [
                    nn.LayerNorm(hidden_dim if i != num_layers - 1 else self.out_dim)
                    for i in range(num_layers)
                ]
            )
        else:
            self.layer_norms = None

        # Add dropout layer
        self.dropout_layer = nn.Dropout(dropout)

    def forward(self, x, edge_index, edge_attr=None, batch=None):
        """
        Forward pass through the GNN model.

        Args:
            x: Node features tensor [num_nodes, input_dim]
            edge_index: Graph connectivity in COO format [2, num_edges]
            edge_attr: Edge features tensor [num_edges, edge_dim] (optional)
            batch: Batch vector [num_nodes] indicating which graph each node belongs to

        Returns:
            Node embeddings [num_nodes, out_dim] or graph embeddings [num_graphs, out_dim]
            depending on whether batch is provided

        Note:
            The forward pass processes the input through all GNN layers with
            optional residual connections and layer normalization.
        """
        # Process through each GNN layer
        for i, conv in enumerate(self.conv_layers):
            # Store input for residual connection
            x_input = x

            # Apply convolution
            if edge_attr is not None and hasattr(conv, "edge_dim"):
                x = conv(x, edge_index, edge_attr)
            else:
                x = conv(x, edge_index)

            # Apply layer normalization if enabled
            if self.layer_norms is not None:
                x = self.layer_norms[i](x)

            # Apply activation function (ReLU)
            x = F.relu(x)

            # Apply dropout
            x = self.dropout_layer(x)

            # Apply residual connection if enabled and dimensions match
            if self.residual and x_input.shape == x.shape:
                x = x + x_input

        # Global pooling if batch information is provided
        if batch is not None:
            x = global_mean_pool(x, batch)

        return x


class GNNClassifier(torch.nn.Module):
    """
    Graph Neural Network classifier for graph-level tasks.

    This class combines GNN layers with a classification head to perform
    graph-level classification tasks such as graph classification or
    graph property prediction.

    Architecture:
        - GNN layers for feature extraction
        - Global pooling for graph-level representations
        - Classification head with configurable layers
        - Dropout and activation functions
    """

    def __init__(
        self, gnn_model, num_classes, hidden_dim=None, num_layers=2, dropout=0.1
    ):
        """
        Initialize the GNN classifier.

        Args:
            gnn_model: Pre-configured GNN model (GNNConv instance)
            num_classes: Number of output classes
            hidden_dim: Hidden dimension for classification head
            num_layers: Number of layers in classification head
            dropout: Dropout probability for regularization
        """
        super(GNNClassifier, self).__init__()

        self.gnn_model = gnn_model
        self.num_classes = num_classes

        # Use GNN output dimension if hidden_dim not specified
        if hidden_dim is None:
            hidden_dim = gnn_model.out_dim

        # Build classification head
        layers = []
        input_dim = gnn_model.out_dim

        for i in range(num_layers - 1):
            layers.extend(
                [nn.Linear(input_dim, hidden_dim), nn.ReLU(), nn.Dropout(dropout)]
            )
            input_dim = hidden_dim

        # Final classification layer
        layers.append(nn.Linear(input_dim, num_classes))

        self.classifier = nn.Sequential(*layers)

    def forward(self, x, edge_index, edge_attr=None, batch=None):
        """
        Forward pass through the GNN classifier.

        Args:
            x: Node features tensor [num_nodes, input_dim]
            edge_index: Graph connectivity [2, num_edges]
            edge_attr: Edge features tensor [num_edges, edge_dim] (optional)
            batch: Batch vector [num_nodes] for graph-level classification

        Returns:
            Logits for each graph [num_graphs, num_classes]

        Note:
            This method requires batch information to perform graph-level
            classification. For node-level tasks, use the GNN model directly.
        """
        # Extract graph-level features using GNN
        graph_features = self.gnn_model(x, edge_index, edge_attr, batch)

        # Apply classification head
        logits = self.classifier(graph_features)

        return logits


class GNNRegressor(torch.nn.Module):
    """
    Graph Neural Network regressor for graph-level regression tasks.

    This class combines GNN layers with a regression head to perform
    graph-level regression tasks such as property prediction or
    continuous value estimation.

    Architecture:
        - GNN layers for feature extraction
        - Global pooling for graph-level representations
        - Regression head with configurable layers
        - Dropout and activation functions
    """

    def __init__(
        self, gnn_model, output_dim=1, hidden_dim=None, num_layers=2, dropout=0.1
    ):
        """
        Initialize the GNN regressor.

        Args:
            gnn_model: Pre-configured GNN model (GNNConv instance)
            output_dim: Dimension of regression output
            hidden_dim: Hidden dimension for regression head
            num_layers: Number of layers in regression head
            dropout: Dropout probability for regularization
        """
        super(GNNRegressor, self).__init__()

        self.gnn_model = gnn_model
        self.output_dim = output_dim

        # Use GNN output dimension if hidden_dim not specified
        if hidden_dim is None:
            hidden_dim = gnn_model.out_dim

        # Build regression head
        layers = []
        input_dim = gnn_model.out_dim

        for i in range(num_layers - 1):
            layers.extend(
                [nn.Linear(input_dim, hidden_dim), nn.ReLU(), nn.Dropout(dropout)]
            )
            input_dim = hidden_dim

        # Final regression layer
        layers.append(nn.Linear(input_dim, output_dim))

        self.regressor = nn.Sequential(*layers)

    def forward(self, x, edge_index, edge_attr=None, batch=None):
        """
        Forward pass through the GNN regressor.

        Args:
            x: Node features tensor [num_nodes, input_dim]
            edge_index: Graph connectivity [2, num_edges]
            edge_attr: Edge features tensor [num_edges, edge_dim] (optional)
            batch: Batch vector [num_nodes] for graph-level regression

        Returns:
            Regression outputs for each graph [num_graphs, output_dim]

        Note:
            This method requires batch information to perform graph-level
            regression. For node-level tasks, use the GNN model directly.
        """
        # Extract graph-level features using GNN
        graph_features = self.gnn_model(x, edge_index, edge_attr, batch)

        # Apply regression head
        outputs = self.regressor(graph_features)

        return outputs
