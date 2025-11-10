"""
Neural network models for benchmarking tasks.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


class ModularArithmeticMLP(nn.Module):
    """
    MLP for modular arithmetic tasks.

    Architecture:
    - Embedding layer for input integers
    - Multiple hidden layers with ReLU activations
    - Output layer with softmax for classification
    """

    def __init__(
        self,
        vocab_size: int,
        embedding_dim: int = 64,
        hidden_dims: list = [128, 128],
        dropout: float = 0.1,
    ):
        super(ModularArithmeticMLP, self).__init__()

        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim

        # Embedding layer for each input
        self.embedding = nn.Embedding(vocab_size, embedding_dim)

        # Build hidden layers
        layers = []
        input_dim = embedding_dim * 2  # Two inputs: a and b

        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(input_dim, hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            input_dim = hidden_dim

        self.hidden_layers = nn.Sequential(*layers)

        # Output layer
        self.output_layer = nn.Linear(input_dim, vocab_size)

    def forward(self, x):
        """
        Args:
            x: Tensor of shape (batch_size, 2) containing integer indices

        Returns:
            Logits of shape (batch_size, vocab_size)
        """
        # x is (batch_size, 2) with integer values
        x = x.long()

        # Embed each input
        emb_a = self.embedding(x[:, 0])  # (batch_size, embedding_dim)
        emb_b = self.embedding(x[:, 1])  # (batch_size, embedding_dim)

        # Concatenate embeddings
        h = torch.cat([emb_a, emb_b], dim=1)  # (batch_size, embedding_dim * 2)

        # Pass through hidden layers
        h = self.hidden_layers(h)

        # Output layer
        logits = self.output_layer(h)

        return logits


class ModularCompositionMLP(nn.Module):
    """
    MLP for modular composition tasks.

    Similar to ModularArithmeticMLP but with flexible input size.
    """

    def __init__(
        self,
        vocab_size: int,
        num_inputs: int = 1,
        embedding_dim: int = 64,
        hidden_dims: list = [128, 128, 128],
        dropout: float = 0.1,
    ):
        super(ModularCompositionMLP, self).__init__()

        self.vocab_size = vocab_size
        self.num_inputs = num_inputs
        self.embedding_dim = embedding_dim

        # Embedding layer
        self.embedding = nn.Embedding(vocab_size, embedding_dim)

        # Build hidden layers
        layers = []
        input_dim = embedding_dim * num_inputs

        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(input_dim, hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            input_dim = hidden_dim

        self.hidden_layers = nn.Sequential(*layers)

        # Output layer
        self.output_layer = nn.Linear(input_dim, vocab_size)

    def forward(self, x):
        """
        Args:
            x: Tensor of shape (batch_size, num_inputs) containing integer indices
               or (batch_size, 1) for single input tasks

        Returns:
            Logits of shape (batch_size, vocab_size)
        """
        x = x.long()

        if len(x.shape) == 1:
            x = x.unsqueeze(1)

        batch_size = x.shape[0]

        # Embed each input and concatenate
        embeddings = []
        for i in range(self.num_inputs):
            emb = self.embedding(x[:, i])
            embeddings.append(emb)

        h = torch.cat(embeddings, dim=1)  # (batch_size, embedding_dim * num_inputs)

        # Pass through hidden layers
        h = self.hidden_layers(h)

        # Output layer
        logits = self.output_layer(h)

        return logits


class SimpleMLP(nn.Module):
    """
    Simple MLP without embeddings for continuous inputs.
    """

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        hidden_dims: list = [128, 128],
        dropout: float = 0.1,
    ):
        super(SimpleMLP, self).__init__()

        layers = []
        prev_dim = input_dim

        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            prev_dim = hidden_dim

        self.hidden_layers = nn.Sequential(*layers)
        self.output_layer = nn.Linear(prev_dim, output_dim)

    def forward(self, x):
        h = self.hidden_layers(x)
        logits = self.output_layer(h)
        return logits


class TransformerModularModel(nn.Module):
    """
    Transformer-based model for modular arithmetic.

    This provides a more challenging benchmark for optimizers.
    """

    def __init__(
        self,
        vocab_size: int,
        embedding_dim: int = 64,
        num_heads: int = 4,
        num_layers: int = 2,
        dropout: float = 0.1,
    ):
        super(TransformerModularModel, self).__init__()

        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.pos_encoding = nn.Parameter(torch.randn(10, embedding_dim))

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embedding_dim,
            nhead=num_heads,
            dim_feedforward=embedding_dim * 4,
            dropout=dropout,
            batch_first=True,
        )

        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.output_layer = nn.Linear(embedding_dim, vocab_size)

    def forward(self, x):
        """
        Args:
            x: Tensor of shape (batch_size, seq_len) with integer indices

        Returns:
            Logits of shape (batch_size, vocab_size)
        """
        x = x.long()
        if len(x.shape) == 1:
            x = x.unsqueeze(1)

        batch_size, seq_len = x.shape

        # Embed inputs
        emb = self.embedding(x)  # (batch_size, seq_len, embedding_dim)

        # Add positional encoding
        emb = emb + self.pos_encoding[:seq_len].unsqueeze(0)

        # Pass through transformer
        h = self.transformer(emb)  # (batch_size, seq_len, embedding_dim)

        # Pool over sequence (mean pooling)
        h = h.mean(dim=1)  # (batch_size, embedding_dim)

        # Output
        logits = self.output_layer(h)

        return logits
