"""
CM-GPT: Causal Modeling GPT for Conceptual Models

This module implements a custom GPT-style architecture specifically designed for
causal modeling tasks in conceptual models. The model combines transformer
architecture with specialized components for processing graph-structured data
and causal relationships.

Key Components:
- Multi-head self-attention with causal masking
- Feed-forward networks with residual connections
- Layer normalization and dropout for regularization
- Configurable architecture for different model sizes

The model is particularly suited for:
- Causal relationship prediction
- Graph-to-text generation
- Sequential modeling of conceptual model elements
- Multi-task learning on graph-structured data

Author: Syed Juned Ali
Email: syed.juned.ali@tuwien.ac.at
"""

import torch
import torch.nn as nn
from glam4cm.settings import device


def weights_init(model):
    """
    Initialize the weights of the model using Xavier uniform initialization.

    This function applies appropriate weight initialization strategies:
    - Xavier uniform for linear layers and embeddings (prevents exploding gradients)
    - Zeros for biases (common practice for transformer models)
    - Ones for layer normalization weights (preserves input distribution)

    Args:
        model: PyTorch model to initialize

    Note:
        Xavier uniform initialization is particularly effective for transformer
        architectures as it maintains variance across layers.
    """
    if isinstance(model, nn.Linear):
        nn.init.xavier_uniform_(model.weight.data)
        if model.bias is not None:
            nn.init.zeros_(model.bias.data)
    elif isinstance(model, nn.Embedding):
        nn.init.xavier_uniform_(model.weight.data)
    elif isinstance(model, nn.LayerNorm):
        nn.init.ones_(model.weight.data)
        nn.init.zeros_(model.bias.data)


class Head(nn.Module):
    """
    Single head of self-attention mechanism.

    This class implements the core attention mechanism used in transformer
    architectures. Each head computes attention scores between all positions
    in the sequence, allowing the model to focus on relevant information.

    Architecture:
        - Key, Query, Value projections
        - Scaled dot-product attention
        - Causal masking for autoregressive generation
        - Dropout for regularization
    """

    def __init__(self, embed_dim, head_size, dropout=0.1):
        """
        Initialize a single attention head.

        Args:
            embed_dim: Input embedding dimension
            head_size: Size of the attention head (typically embed_dim // num_heads)
            dropout: Dropout probability for regularization
        """
        super().__init__()
        self.key = nn.Linear(embed_dim, head_size, bias=False)
        self.query = nn.Linear(embed_dim, head_size, bias=False)
        self.value = nn.Linear(embed_dim, head_size, bias=False)

        # Causal mask for autoregressive generation (lower triangular)
        self.register_buffer("tril", torch.tril(torch.ones(head_size, head_size)))
        self.softmax = nn.Softmax(dim=-1)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, attention_mask):
        """
        Forward pass through the attention head.

        Args:
            x: Input tensor of shape [batch_size, seq_len, embed_dim]
            attention_mask: Boolean mask of shape [batch_size, seq_len] where
                           True indicates valid positions and False indicates
                           padding or invalid positions

        Returns:
            Output tensor of shape [batch_size, seq_len, head_size]

        Note:
            The attention mechanism computes relationships between all positions
            while respecting the causal mask and attention mask constraints.
        """
        _, _, C = x.shape

        # Project inputs to key, query, and value spaces
        k = self.key(x)
        q = self.query(x)

        # Compute attention scores with scaling factor
        # The scaling factor C**-0.5 prevents softmax saturation
        wei = q @ k.transpose(-2, -1) * C**-0.5

        # Apply attention mask (set masked positions to -inf)
        wei = wei.masked_fill((attention_mask.unsqueeze(1) == 0), float("-inf"))

        # Apply softmax to get attention probabilities
        wei = self.softmax(wei)
        wei = self.dropout(wei)

        # Compute weighted aggregation of values
        v = self.value(x)
        out = wei @ v
        return out


class MultiHeadAttention(nn.Module):
    """
    Multi-head self-attention mechanism.

    This module implements the standard multi-head attention used in transformer
    architectures. It splits the embedding dimension into multiple heads,
    allowing the model to attend to different types of relationships
    simultaneously.

    Architecture:
        - Multiple parallel attention heads
        - Concatenation of head outputs
        - Final projection layer
        - Dropout for regularization
    """

    def __init__(self, embed_dim, num_heads, dropout=0.1):
        """
        Initialize multi-head attention.

        Args:
            embed_dim: Input embedding dimension
            num_heads: Number of attention heads (must divide embed_dim evenly)
            dropout: Dropout probability for regularization
        """
        super().__init__()
        assert embed_dim % num_heads == 0, "embed_dim must be divisible by num_heads"

        head_size = embed_dim // num_heads
        self.heads = nn.ModuleList(
            [Head(embed_dim, head_size) for _ in range(num_heads)]
        )
        self.proj = nn.Linear(embed_dim, embed_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, attn_mask):
        """
        Forward pass through multi-head attention.

        Args:
            x: Input tensor of shape [batch_size, seq_len, embed_dim]
            attn_mask: Attention mask for valid positions

        Returns:
            Output tensor of shape [batch_size, seq_len, embed_dim]
        """
        # Process each attention head in parallel
        out = torch.cat([h(x, attn_mask) for h in self.heads], dim=-1)

        # Apply final projection and dropout
        out = self.dropout(self.proj(out))
        return out


class FeedFoward(nn.Module):
    """
    Feed-forward network with configurable dimensions.

    This module implements the standard feed-forward network used in transformer
    architectures. It consists of two linear transformations with a non-linearity
    in between, providing the model with additional computational capacity.

    Architecture:
        - First linear layer (expansion)
        - GELU activation function
        - Second linear layer (projection)
        - Dropout for regularization
    """

    def __init__(self, input_dim, embed_dim=None, num_classes=None, dropout=0.1):
        """
        Initialize feed-forward network.

        Args:
            input_dim: Input dimension
            embed_dim: Hidden dimension (defaults to input_dim if None)
            num_classes: Output dimension (defaults to embed_dim if None)
            dropout: Dropout probability for regularization
        """
        super().__init__()

        # Set default dimensions if not specified
        if num_classes is None:
            num_classes = input_dim if embed_dim is None else embed_dim
        if embed_dim is None:
            embed_dim = input_dim

        # Build the feed-forward network
        self.net = nn.Sequential(
            nn.Linear(input_dim, embed_dim),
            nn.GELU(),  # GELU is often preferred over ReLU in transformers
            nn.Dropout(dropout),
            nn.Linear(embed_dim, num_classes),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        """
        Forward pass through the feed-forward network.

        Args:
            x: Input tensor

        Returns:
            Output tensor after feed-forward transformation
        """
        return self.net(x)


class Block(nn.Module):
    """
    Transformer block combining attention and feed-forward components.

    This module implements a complete transformer block with:
    - Multi-head self-attention
    - Feed-forward network
    - Layer normalization
    - Residual connections
    - Dropout for regularization

    The block follows the standard transformer architecture and can be stacked
    to create deep transformer models.
    """

    def __init__(self, embed_dim, num_heads, dropout=0.1):
        """
        Initialize transformer block.

        Args:
            embed_dim: Embedding dimension
            num_heads: Number of attention heads
            dropout: Dropout probability for regularization
        """
        super().__init__()

        # Multi-head attention layer
        self.sa = MultiHeadAttention(embed_dim, num_heads, dropout)

        # Feed-forward network
        self.ffwd = FeedFoward(embed_dim, dropout=dropout)

        # Layer normalization layers
        self.ln1 = nn.LayerNorm(embed_dim)
        self.ln2 = nn.LayerNorm(embed_dim)

        # Dropout for regularization
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, attention_mask):
        """
        Forward pass through the transformer block.

        Args:
            x: Input tensor of shape [batch_size, seq_len, embed_dim]
            attention_mask: Attention mask for valid positions

        Returns:
            Output tensor after transformer block processing
        """
        # Self-attention with residual connection and layer normalization
        x = x + self.dropout(self.sa(self.ln1(x), attention_mask))

        # Feed-forward with residual connection and layer normalization
        x = x + self.dropout(self.ffwd(self.ln2(x)))

        return x


class CMGPT(nn.Module):
    """
    CM-GPT: Causal Modeling GPT for Conceptual Models.

    This is the main model class that implements a complete GPT-style architecture
    for causal modeling tasks. The model processes sequences of tokens and can
    be used for various downstream tasks including:

    - Causal relationship prediction
    - Graph-to-text generation
    - Sequential modeling of conceptual model elements
    - Multi-task learning on graph-structured data

    Architecture:
        - Token embeddings
        - Positional embeddings
        - Stack of transformer blocks
        - Task-specific output head
        - Configurable model size and architecture
    """

    def __init__(
        self,
        vocab_size,
        embed_dim,
        num_heads,
        num_layers,
        num_classes=None,
        dropout=0.1,
        max_seq_len=1024,
    ):
        """
        Initialize CM-GPT model.

        Args:
            vocab_size: Size of the vocabulary
            embed_dim: Embedding dimension
            num_heads: Number of attention heads
            num_layers: Number of transformer blocks
            num_classes: Number of output classes (for classification tasks)
            dropout: Dropout probability for regularization
            max_seq_len: Maximum sequence length for positional embeddings
        """
        super().__init__()

        # Model configuration
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.max_seq_len = max_seq_len

        # Token and positional embeddings
        self.token_embedding_table = nn.Embedding(vocab_size, embed_dim)
        self.position_embedding_table = nn.Embedding(max_seq_len, embed_dim)

        # Stack of transformer blocks
        self.blocks = nn.ModuleList(
            [Block(embed_dim, num_heads, dropout) for _ in range(num_layers)]
        )

        # Final layer normalization
        self.ln_f = nn.LayerNorm(embed_dim)

        # Output head (configurable for different tasks)
        if num_classes is not None:
            self.output_head = nn.Linear(embed_dim, num_classes)
        else:
            self.output_head = None

        # Initialize model weights
        self.apply(weights_init)

    def forward(self, idx, attention_mask=None, targets=None):
        """
        Forward pass through the CM-GPT model.

        Args:
            idx: Input token indices of shape [batch_size, seq_len]
            attention_mask: Boolean mask for valid positions [batch_size, seq_len]
            targets: Target values for supervised learning (optional)

        Returns:
            If targets provided: (logits, loss)
            If no targets: logits only

        Note:
            The model can be used in both training (with targets) and
            inference (without targets) modes.
        """
        batch_size, seq_len = idx.shape

        # Create attention mask if not provided
        if attention_mask is None:
            attention_mask = torch.ones(
                batch_size, seq_len, dtype=torch.bool, device=idx.device
            )

        # Get token and positional embeddings
        tok_emb = self.token_embedding_table(idx)
        pos_emb = self.position_embedding_table(
            torch.arange(seq_len, device=idx.device)
        )

        # Combine embeddings
        x = tok_emb + pos_emb

        # Process through transformer blocks
        for block in self.blocks:
            x = block(x, attention_mask)

        # Final layer normalization
        x = self.ln_f(x)

        # Apply output head if available
        if self.output_head is not None:
            logits = self.output_head(x)
        else:
            logits = x

        # Compute loss if targets are provided
        if targets is not None:
            # For classification tasks, use cross-entropy loss
            if len(logits.shape) == 3:  # [batch_size, seq_len, num_classes]
                logits = logits.view(-1, logits.size(-1))
                targets = targets.view(-1)

            loss = nn.functional.cross_entropy(logits, targets)
            return logits, loss

        return logits

    def generate(
        self,
        idx,
        max_new_tokens,
        attention_mask=None,
        temperature=1.0,
        do_sample=False,
        top_k=None,
    ):
        """
        Generate new tokens autoregressively.

        Args:
            idx: Starting sequence of token indices [batch_size, seq_len]
            max_new_tokens: Maximum number of new tokens to generate
            attention_mask: Attention mask for the input sequence
            temperature: Sampling temperature (higher = more random)
            do_sample: Whether to use sampling or greedy decoding
            top_k: Top-k sampling parameter

        Returns:
            Generated sequence with new tokens appended

        Note:
            This method implements autoregressive generation, where each new
            token is predicted based on all previous tokens.
        """
        for _ in range(max_new_tokens):
            # Get predictions for the next token
            logits = self.forward(idx, attention_mask)
            logits = logits[:, -1, :] / temperature

            # Apply top-k filtering if specified
            if top_k is not None:
                v, _ = torch.topk(logits, top_k)
                logits[logits < v[:, [-1]]] = -float("Inf")

            # Sample or select the most likely token
            if do_sample:
                probs = nn.functional.softmax(logits, dim=-1)
                idx_next = torch.multinomial(probs, num_samples=1)
            else:
                idx_next = torch.argmax(logits, dim=-1, keepdim=True)

            # Append the new token to the sequence
            idx = torch.cat((idx, idx_next), dim=1)

            # Update attention mask if provided
            if attention_mask is not None:
                new_mask = torch.ones(
                    idx.shape[0], 1, dtype=torch.bool, device=idx.device
                )
                attention_mask = torch.cat((attention_mask, new_mask), dim=1)

        return idx
