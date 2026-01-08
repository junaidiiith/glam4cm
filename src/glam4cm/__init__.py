"""
GLAM4CM: Graph Language Models for Conceptual Modeling

A comprehensive framework that combines Graph Neural Networks (GNNs) and Language
Models (LMs) to tackle various downstream tasks on conceptual models. This
framework bridges the gap between structured graph representations and natural
language understanding, making it particularly suitable for conceptual modeling
domains like UML, Ecore, ArchiMate, and BPMN.

Key Features:
- Multi-modal architecture supporting both GNNs and LMs
- Comprehensive downstream task support (classification, prediction, generation)
- Graph-text bidirectional conversion capabilities
- Support for multiple conceptual modeling languages
- Advanced model architectures including CM-GPT and attention-based GNNs

Architecture Overview:
    ┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
    │   Data Loading  │    │  Model Layer    │    │ Downstream      │
    │   & Processing  │───▶│  (GNNs + LMs)   │───▶│ Tasks          │
    └─────────────────┘    └─────────────────┘    └─────────────────┘
           │                       │                       │
           ▼                       ▼                       ▼
    ┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
    │ Graph2Text      │    │ Training &      │    │ Evaluation &    │
    │ Conversion      │    │ Optimization    │    │ Results         │
    └─────────────────┘    └─────────────────┘    └─────────────────┘

Supported Conceptual Models:
- UML (Unified Modeling Language)
- Ecore (Eclipse Modeling Framework)
- ArchiMate (Enterprise Architecture)
- BPMN (Business Process Model and Notation)
- OntoUML (Ontological UML Extensions)

Supported Models:
- Graph Neural Networks: GCN, GraphSAGE, GAT, GIN, GraphConv
- Language Models: BERT variants, ModernBERT, CM-GPT
- Traditional ML: TF-IDF, Word2Vec, FastText

Downstream Tasks:
- Node Classification: Classify nodes in conceptual models
- Edge Classification: Classify relationships between model elements
- Link Prediction: Predict missing connections in models
- Graph Classification: Categorize entire conceptual models
- Text Classification: Process textual descriptions
- Causal Modeling: Predict causal relationships with CM-GPT

Quick Start:
    # Install the package
    pip install glam4cm
    
    # Run a BERT node classification task
    python -m glam4cm.run --task_id 3 \
        --model_name bert-base-uncased \
        --dataset_path your_dataset.json \
        --output_dir results/bert_node_cls
    
    # Run a GNN graph classification task
    python -m glam4cm.run --task_id 6 \
        --model_name GCNConv \
        --hidden_dim 128 \
        --num_layers 3 \
        --dataset_path your_dataset.json \
        --output_dir results/gnn_graph_cls

Package Structure:
    glam4cm/
    ├── data_loading/          # Dataset management and preprocessing
    ├── downstream_tasks/      # Task-specific implementations
    ├── embeddings/            # Text embedding models
    ├── encoding/              # Graph encoding and representation
    ├── graph2str/             # Graph-to-text conversion
    ├── lang2graph/            # Text-to-graph conversion
    ├── models/                # Neural network architectures
    ├── tokenization/          # Text tokenization utilities
    └── trainers/              # Training and optimization

For more information, see the README.md file or visit:
https://github.com/junaidiiith/glam4cm

Author: Syed Juned Ali
Email: syed.juned.ali@tuwien.ac.at
Institution: TU Wien
License: MIT
"""

import warnings

# Suppress Pydantic serializer warnings for cleaner output
warnings.filterwarnings(
    "ignore",
    message="Pydantic serializer warnings:",
    category=UserWarning,
    module="pydantic.main",
)

# Package version
__version__ = "1.0.0"

# Package metadata
__author__ = "Syed Juned Ali"
__email__ = "syed.juned.ali@tuwien.ac.at"
__institution__ = "TU Wien"
__license__ = "MIT"
__url__ = "https://github.com/junaidiiith/glam4cm"

# Package description
__description__ = "Graph Language Models for Conceptual Modeling"
__long_description__ = __doc__

# Supported conceptual modeling languages
SUPPORTED_LANGUAGES = [
    "UML",  # Unified Modeling Language
    "Ecore",  # Eclipse Modeling Framework
    "ArchiMate",  # Enterprise Architecture
    "BPMN",  # Business Process Model and Notation
    "OntoUML",  # Ontological UML Extensions
]

# Supported model types
SUPPORTED_MODELS = {
    "gnn": ["GCNConv", "GraphConv", "GATConv", "SAGEConv", "GINConv", "GATv2Conv"],
    "lm": ["bert-base-uncased", "answerdotai/ModernBERT-base", "CM-GPT"],
    "traditional": ["tfidf", "word2vec", "fasttext"],
}

# Supported downstream tasks
SUPPORTED_TASKS = {
    "classification": ["node_cls", "edge_cls", "graph_cls"],
    "prediction": ["link_prediction"],
    "generation": ["text_generation", "causal_modeling"],
}

# Package imports for easy access
try:
    from .run import main
    from .models.cmgpt import CMGPT
    from .models.gnn_layers import GNNConv, GNNClassifier, GNNRegressor
    from .graph2str.ontouml import get_node_text_triples
    from .lang2graph.ecore import EcoreNxG

    __all__ = [
        "main",
        "CMGPT",
        "GNNConv",
        "GNNClassifier",
        "GNNRegressor",
        "get_node_text_triples",
        "EcoreNxG",
    ]

except ImportError as e:
    # Handle import errors gracefully for development
    warnings.warn(f"Some GLAM4CM components could not be imported: {e}")
    __all__ = []
