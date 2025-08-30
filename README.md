# GLAM4CM: Graph Language Models for Conceptual Modeling

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-1.8+-red.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

**GLAM4CM** (Graph Language Models for Conceptual Modeling) is a comprehensive framework that combines Graph Neural Networks (GNNs) and Language Models (LMs) to tackle various downstream tasks on conceptual models. This framework is designed to bridge the gap between structured graph representations and natural language understanding, making it particularly suitable for conceptual modeling domains like UML, Ecore, ArchiMate, and BPMN.

## 🚀 Features

### **Multi-Modal Architecture**

- **Graph Neural Networks**: Support for GCN, GraphSAGE, GAT, GIN, and GraphConv
- **Language Models**: BERT-based models with fine-tuning capabilities
- **Hybrid Models**: CM-GPT for causal modeling tasks
- **Traditional ML**: TF-IDF and Word2Vec baselines

### **Downstream Tasks**

- **Node Classification**: Classify nodes in conceptual models
- **Edge Classification**: Classify relationships between model elements
- **Link Prediction**: Predict missing connections in models
- **Graph Classification**: Categorize entire conceptual models
- **Text Classification**: Process textual descriptions of model elements

### **Conceptual Model Support**

- **UML Models**: Class diagrams, sequence diagrams, use case diagrams
- **Ecore Models**: EMF/Ecore metamodels
- **ArchiMate Models**: Enterprise architecture models
- **BPMN Models**: Business process models
- **OntoUML Models**: Ontological UML extensions

### **Advanced Capabilities**

- **Graph-to-Text Conversion**: Transform graph structures to natural language
- **Text-to-Graph Conversion**: Parse textual descriptions into graph representations
- **Multi-hop Reasoning**: Support for k-step neighborhood exploration
- **Attention Mechanisms**: Self-attention and graph attention networks
- **Residual Connections**: Deep network architectures with skip connections

## 🏗️ Architecture

```
GLAM4CM/
├── 📊 Data Loading & Processing
│   ├── Graph dataset management
│   ├── Text encoding and tokenization
│   └── Multi-format data support
├── 🧠 Model Architectures
│   ├── GNN layers (GCN, GAT, SAGE, GIN)
│   ├── Transformer-based models (BERT, CM-GPT)
│   └── Hybrid architectures
├── 🎯 Downstream Tasks
│   ├── Node/Edge/Graph classification
│   ├── Link prediction
│   └── Text classification
├── 🔄 Graph-Text Conversion
│   ├── Graph2Str: Convert graphs to text
│   └── Lang2Graph: Parse text to graphs
└── 🚀 Training & Evaluation
    ├── Multi-task training pipelines
    ├── Comprehensive evaluation metrics
    └── Experimental result analysis
```

## 📦 Installation

### Prerequisites

- Python 3.8 or higher
- PyTorch 1.8+
- CUDA-compatible GPU (optional, for acceleration)

### Quick Install

```bash
# Clone the repository
git clone https://github.com/junaidiiith/glam4cm.git
cd glam4cm

# Install dependencies
pip install -r requirements.txt

# Install the package
pip install -e .
```

### Manual Installation

```bash
# Install core dependencies
pip install torch torch-geometric transformers sentence-transformers
pip install networkx pandas scikit-learn tqdm tensorboardX

# Install optional dependencies
pip install fasttext xmltodict
```

## 🚀 Quick Start

### 1. Create a Dataset

```bash
python -m glam4cm.run --task_id 0 --help
```

### 2. Train a BERT Node Classifier

```bash
python -m glam4cm.run --task_id 3 \
    --model_name bert-base-uncased \
    --dataset_path your_dataset.json \
    --output_dir results/bert_node_cls
```

### 3. Train a GNN Graph Classifier

```bash
python -m glam4cm.run --task_id 6 \
    --model_name GCNConv \
    --hidden_dim 128 \
    --num_layers 3 \
    --dataset_path your_dataset.json \
    --output_dir results/gnn_graph_cls
```

### 4. Train CM-GPT for Causal Modeling

```bash
python -m glam4cm.run --task_id 10 \
    --embed_dim 256 \
    --num_heads 8 \
    --num_layers 6 \
    --dataset_path your_dataset.json \
    --output_dir results/cm_gpt
```

## 📚 Usage Examples

### Basic Usage

```python
from glam4cm.run import main
import sys

# Set up arguments for BERT node classification
sys.argv = [
    'run.py',
    '--task_id', '3',
    '--model_name', 'bert-base-uncased',
    '--dataset_path', 'datasets/ecore_models.json',
    '--output_dir', 'results/bert_nc',
    '--batch_size', '16',
    '--learning_rate', '2e-5',
    '--num_epochs', '10'
]

# Run the task
main()
```

### Custom Model Configuration

```python
from glam4cm.models.gnn_layers import GNNConv

# Create a custom GNN model
gnn_model = GNNConv(
    model_name='GATConv',
    input_dim=128,
    hidden_dim=256,
    out_dim=64,
    num_layers=4,
    num_heads=8,
    residual=True,
    l_norm=True,
    dropout=0.1,
    aggregation='mean'
)
```

### Graph-to-Text Conversion

```python
from glam4cm.graph2str.ontouml import get_node_text_triples

# Convert graph nodes to text descriptions
node_descriptions = get_node_text_triples(
    graph,
    distance=2,
    only_name=False
)
```

## 🔧 Configuration

### Model Settings

Key configuration options in `src/glam4cm/settings.py`:

```python
# BERT Models
BERT_MODEL = 'bert-base-uncased'
MODERN_BERT = 'answerdotai/ModernBERT-base'

# Word2Vec Configuration
W2V_CONFIG = dict(
    epoch=100,
    dim=128,
    ws=5,
    minCount=1,
    thread=4,
    model='skipgram'
)

# Task Types
NODE_CLS_TASK = 'node_cls'
EDGE_CLS_TASK = 'edge_cls'
LINK_PRED_TASK = 'lp'
GRAPH_CLS_TASK = 'graph_cls'
```

### Training Parameters

Common training arguments available across all tasks:

```bash
--batch_size          # Batch size for training
--learning_rate       # Learning rate for optimization
--num_epochs          # Number of training epochs
--weight_decay        # Weight decay for regularization
--dropout            # Dropout probability
--hidden_dim         # Hidden dimension size
--num_layers         # Number of model layers
```

## 📊 Supported Models

### Graph Neural Networks

| Model     | Multi-Head Support | Aggregation Methods |
| --------- | ------------------ | ------------------- |
| GCNConv   | ❌                 | Mean, Sum, Max, Mul |
| GraphConv | ❌                 | Mean, Sum, Max, Mul |
| GATConv   | ✅                 | Mean, Sum, Max, Mul |
| SAGEConv  | ❌                 | Mean, Sum, Max, Mul |
| GINConv   | ❌                 | Mean, Sum, Max, Mul |
| GATv2Conv | ✅                 | Mean, Sum, Max, Mul |

### Language Models

- **BERT Variants**: bert-base-uncased, ModernBERT
- **Custom Models**: CM-GPT with configurable architecture
- **Embedding Models**: Sentence transformers, FastText, Word2Vec

### Aggregation Methods

- **Global Pooling**: Sum, Mean, Max
- **Node Aggregation**: Mean, Sum, Max, Mul
- **Edge Aggregation**: Mean, Sum, Max, Mul

## 🧪 Experimental Results

The framework includes comprehensive experimental results in the `experimental_results/` directory:

- **Overall Results**: Performance metrics across all tasks
- **Statistical Tests**: Significance testing for research questions
- **Baseline Comparisons**: FTLM vs. GNN vs. traditional baselines
- **Configuration Analysis**: Impact of different hyperparameters

## 🔬 Research Applications

GLAM4CM is designed for research in:

- **Conceptual Modeling**: UML, Ecore, ArchiMate, BPMN
- **Graph Representation Learning**: Node, edge, and graph embeddings
- **Multi-Modal Learning**: Combining structured and textual data
- **Domain-Specific NLP**: Technical documentation processing
- **Software Engineering**: Model-driven development, reverse engineering

## 🤝 Contributing

We welcome contributions! Please see our contributing guidelines:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

### Development Setup

```bash
# Install development dependencies
pip install -e ".[dev]"

# Run tests
pytest

# Format code
black src/
```

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- **PyTorch Geometric** for GNN implementations
- **Transformers** library for BERT-based models
- **NetworkX** for graph operations
- **Research Community** for conceptual modeling datasets

## 📞 Contact

- **Author**: Syed Juned Ali
- **Email**: syed.juned.ali@tuwien.ac.at
- **Institution**: TU Wien
- **GitHub**: [@junaidiiith](https://github.com/junaidiiith)

## 📚 Citation

If you use GLAM4CM in your research, please cite:

```bibtex
@software{glam4cm2024,
  title={GLAM4CM: Graph Language Models for Conceptual Modeling},
  author={Ali, Syed Juned},
  year={2024},
  url={https://github.com/junaidiiith/glam4cm}
}
```

---

**⭐ Star this repository if you find it useful!**

**🔗 Related Projects**: [Conceptual Modeling Tools](https://github.com/topics/conceptual-modeling), [Graph Neural Networks](https://github.com/topics/graph-neural-networks), [Language Models](https://github.com/topics/language-models)
