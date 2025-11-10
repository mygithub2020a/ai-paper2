"""
Neural network models for modular arithmetic tasks.

Includes:
- Simple MLPs (2-4 layers, 6-8 layers)
- Tiny transformer/MLP-mixer
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List


class SimpleMLP(nn.Module):
    """Simple fully-connected network."""

    def __init__(
        self,
        input_dim: int,
        hidden_dims: List[int],
        output_dim: int,
        activation: str = 'relu',
        dropout: float = 0.0,
        batch_norm: bool = False,
    ):
        """
        Args:
            input_dim: Input dimension
            hidden_dims: List of hidden layer dimensions
            output_dim: Output dimension
            activation: Activation function ('relu', 'gelu', 'tanh')
            dropout: Dropout probability
            batch_norm: Whether to use batch normalization
        """
        super().__init__()

        self.input_dim = input_dim
        self.output_dim = output_dim

        # Activation function
        if activation == 'relu':
            self.activation = nn.ReLU()
        elif activation == 'gelu':
            self.activation = nn.GELU()
        elif activation == 'tanh':
            self.activation = nn.Tanh()
        else:
            raise ValueError(f"Unknown activation: {activation}")

        # Build layers
        layers = []
        prev_dim = input_dim

        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            if batch_norm:
                layers.append(nn.BatchNorm1d(hidden_dim))
            layers.append(self.activation)
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            prev_dim = hidden_dim

        # Output layer
        layers.append(nn.Linear(prev_dim, output_dim))

        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)


class ResidualBlock(nn.Module):
    """Residual block with optional batch norm."""

    def __init__(self, dim: int, dropout: float = 0.0, batch_norm: bool = False):
        super().__init__()
        layers = [nn.Linear(dim, dim)]
        if batch_norm:
            layers.append(nn.BatchNorm1d(dim))
        layers.append(nn.ReLU())
        if dropout > 0:
            layers.append(nn.Dropout(dropout))
        layers.append(nn.Linear(dim, dim))
        if batch_norm:
            layers.append(nn.BatchNorm1d(dim))

        self.block = nn.Sequential(*layers)
        self.activation = nn.ReLU()

    def forward(self, x):
        return self.activation(x + self.block(x))


class ResidualMLP(nn.Module):
    """MLP with residual connections."""

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
        n_blocks: int = 4,
        dropout: float = 0.0,
        batch_norm: bool = False,
    ):
        super().__init__()

        self.input_proj = nn.Linear(input_dim, hidden_dim)
        self.blocks = nn.ModuleList([
            ResidualBlock(hidden_dim, dropout, batch_norm)
            for _ in range(n_blocks)
        ])
        self.output_proj = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        x = self.input_proj(x)
        for block in self.blocks:
            x = block(x)
        return self.output_proj(x)


class MLPMixer(nn.Module):
    """Simplified MLP-Mixer architecture."""

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
        n_layers: int = 4,
        dropout: float = 0.0,
    ):
        super().__init__()

        self.input_proj = nn.Linear(input_dim, hidden_dim)

        # Mixer layers
        self.mixer_layers = nn.ModuleList()
        for _ in range(n_layers):
            self.mixer_layers.append(nn.ModuleDict({
                'norm1': nn.LayerNorm(hidden_dim),
                'mlp1': nn.Sequential(
                    nn.Linear(hidden_dim, hidden_dim * 4),
                    nn.GELU(),
                    nn.Dropout(dropout),
                    nn.Linear(hidden_dim * 4, hidden_dim),
                    nn.Dropout(dropout),
                ),
                'norm2': nn.LayerNorm(hidden_dim),
                'mlp2': nn.Sequential(
                    nn.Linear(hidden_dim, hidden_dim * 4),
                    nn.GELU(),
                    nn.Dropout(dropout),
                    nn.Linear(hidden_dim * 4, hidden_dim),
                    nn.Dropout(dropout),
                ),
            }))

        self.output_proj = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, x):
        x = self.input_proj(x)

        for layer in self.mixer_layers:
            # Token mixing
            x = x + layer['mlp1'](layer['norm1'](x))
            # Channel mixing
            x = x + layer['mlp2'](layer['norm2'](x))

        return self.output_proj(x)


def get_model(
    model_type: str,
    input_dim: int,
    output_dim: int,
    hidden_dim: int = 128,
    n_layers: int = 4,
    **kwargs
) -> nn.Module:
    """
    Factory function to create models.

    Args:
        model_type: 'mlp_small', 'mlp_medium', 'mlp_large', 'residual', 'mixer'
        input_dim: Input dimension
        output_dim: Output dimension
        hidden_dim: Hidden dimension
        n_layers: Number of layers
        **kwargs: Additional model-specific arguments

    Returns:
        Model instance
    """
    if model_type == 'mlp_small':
        return SimpleMLP(
            input_dim=input_dim,
            hidden_dims=[hidden_dim] * 2,
            output_dim=output_dim,
            **kwargs
        )
    elif model_type == 'mlp_medium':
        return SimpleMLP(
            input_dim=input_dim,
            hidden_dims=[hidden_dim] * 4,
            output_dim=output_dim,
            **kwargs
        )
    elif model_type == 'mlp_large':
        return SimpleMLP(
            input_dim=input_dim,
            hidden_dims=[hidden_dim] * n_layers,
            output_dim=output_dim,
            **kwargs
        )
    elif model_type == 'residual':
        return ResidualMLP(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            output_dim=output_dim,
            n_blocks=n_layers,
            **kwargs
        )
    elif model_type == 'mixer':
        return MLPMixer(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            output_dim=output_dim,
            n_layers=n_layers,
            **kwargs
        )
    else:
        raise ValueError(f"Unknown model_type: {model_type}")


if __name__ == '__main__':
    # Test models
    batch_size = 32
    input_dim = 16
    output_dim = 97

    for model_type in ['mlp_small', 'mlp_medium', 'residual', 'mixer']:
        model = get_model(model_type, input_dim, output_dim, hidden_dim=128)
        x = torch.randn(batch_size, input_dim)
        y = model(x)
        print(f"{model_type}: input {x.shape} -> output {y.shape}")

        # Count parameters
        n_params = sum(p.numel() for p in model.parameters())
        print(f"  Parameters: {n_params:,}")
