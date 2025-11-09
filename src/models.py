import torch
import torch.nn as nn

class MLP(nn.Module):
    def __init__(self, input_dim, output_dim, num_hidden_layers, hidden_dim, use_embedding=False, embedding_dim=None):
        super().__init__()
        self.use_embedding = use_embedding
        if self.use_embedding:
            # For embedding, input_dim is the number of embeddings (e.g., vocabulary size)
            self.embedding = nn.Embedding(input_dim, embedding_dim)
            current_dim = embedding_dim
        else:
            current_dim = input_dim

        layers = []
        layers.append(nn.Linear(current_dim, hidden_dim))
        layers.append(nn.ReLU())

        for _ in range(num_hidden_layers - 1):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            layers.append(nn.ReLU())

        layers.append(nn.Linear(hidden_dim, output_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        if self.use_embedding:
            # The input x is expected to be long integers for embedding
            x = self.embedding(x.squeeze(-1))
        else:
            # The input x is expected to be float
            x = x.float()
        return self.net(x)

def create_modular_arithmetic_model(p=97, hidden_units=128):
    """
    Creates a 2-layer MLP for the modular arithmetic task.
    A 2-layer MLP has one hidden layer.
    """
    # The input is an integer from 0 to p-1. We use an embedding layer.
    # The output is a probability distribution over p classes.
    return MLP(input_dim=p, output_dim=p, num_hidden_layers=1, hidden_dim=hidden_units, use_embedding=True, embedding_dim=32)

def create_modular_composition_model(p=97, hidden_units=256):
    """
    Creates a 3-layer MLP for the modular composition task.
    A 3-layer MLP has two hidden layers.
    """
    return MLP(input_dim=p, output_dim=p, num_hidden_layers=2, hidden_dim=hidden_units, use_embedding=True, embedding_dim=32)

def create_sparse_parity_model(n_bits=16, hidden_units=128):
    """
    Creates a 2-layer MLP for the sparse parity task.
    """
    # Input is a vector of n_bits {0, 1}. Output is a logit for 0 or 1.
    return MLP(input_dim=n_bits, output_dim=2, num_hidden_layers=1, hidden_dim=hidden_units)
