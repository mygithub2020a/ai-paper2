"""
Neural network architectures for policy-value networks.

Implements networks similar to AlphaZero for board games.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple


class PolicyValueNetwork(nn.Module):
    """
    Simple MLP-based policy-value network.

    Args:
        input_dim: Input state dimension
        action_dim: Number of possible actions
        hidden_dims: List of hidden layer dimensions
    """

    def __init__(
        self,
        input_dim: int,
        action_dim: int,
        hidden_dims: list = [256, 256],
    ):
        super(PolicyValueNetwork, self).__init__()

        self.input_dim = input_dim
        self.action_dim = action_dim

        # Shared trunk
        layers = []
        prev_dim = input_dim
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.ReLU())
            prev_dim = hidden_dim

        self.trunk = nn.Sequential(*layers)

        # Policy head
        self.policy_head = nn.Sequential(
            nn.Linear(prev_dim, action_dim),
        )

        # Value head
        self.value_head = nn.Sequential(
            nn.Linear(prev_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Tanh(),
        )

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass.

        Args:
            x: Input state tensor of shape (batch_size, input_dim)

        Returns:
            policy_logits: Action logits of shape (batch_size, action_dim)
            value: State value of shape (batch_size, 1)
        """
        features = self.trunk(x)
        policy_logits = self.policy_head(features)
        value = self.value_head(features)
        return policy_logits, value


class ResBlock(nn.Module):
    """Residual block for ResNet."""

    def __init__(self, channels: int):
        super(ResBlock, self).__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += residual
        out = F.relu(out)
        return out


class ResNetPolicyValue(nn.Module):
    """
    ResNet-based policy-value network (AlphaZero-style).

    Args:
        board_size: Size of the game board (e.g., 8 for chess, 19 for Go)
        action_dim: Number of possible actions
        num_channels: Number of channels in conv layers
        num_res_blocks: Number of residual blocks
    """

    def __init__(
        self,
        board_size: int,
        action_dim: int,
        num_channels: int = 256,
        num_res_blocks: int = 19,
        input_channels: int = 17,
    ):
        super(ResNetPolicyValue, self).__init__()

        self.board_size = board_size
        self.action_dim = action_dim

        # Initial conv layer
        self.conv_input = nn.Conv2d(
            input_channels, num_channels, kernel_size=3, padding=1
        )
        self.bn_input = nn.BatchNorm2d(num_channels)

        # Residual tower
        self.res_blocks = nn.ModuleList(
            [ResBlock(num_channels) for _ in range(num_res_blocks)]
        )

        # Policy head
        self.policy_conv = nn.Conv2d(num_channels, 32, kernel_size=1)
        self.policy_bn = nn.BatchNorm2d(32)
        self.policy_fc = nn.Linear(32 * board_size * board_size, action_dim)

        # Value head
        self.value_conv = nn.Conv2d(num_channels, 32, kernel_size=1)
        self.value_bn = nn.BatchNorm2d(32)
        self.value_fc1 = nn.Linear(32 * board_size * board_size, 256)
        self.value_fc2 = nn.Linear(256, 1)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass.

        Args:
            x: Input state tensor of shape (batch_size, input_channels, board_size, board_size)

        Returns:
            policy_logits: Action logits of shape (batch_size, action_dim)
            value: State value of shape (batch_size, 1)
        """
        # Initial conv
        out = F.relu(self.bn_input(self.conv_input(x)))

        # Residual tower
        for res_block in self.res_blocks:
            out = res_block(out)

        # Policy head
        policy = F.relu(self.policy_bn(self.policy_conv(out)))
        policy = policy.view(policy.size(0), -1)
        policy_logits = self.policy_fc(policy)

        # Value head
        value = F.relu(self.value_bn(self.value_conv(out)))
        value = value.view(value.size(0), -1)
        value = F.relu(self.value_fc1(value))
        value = torch.tanh(self.value_fc2(value))

        return policy_logits, value


class QuantumInspiredPolicyValue(nn.Module):
    """
    Quantum-inspired policy-value network with superposition-like features.

    This network incorporates quantum-inspired elements:
    - Superposition of features
    - Entanglement-like correlations between policy and value
    - Measurement-inspired probabilistic outputs
    """

    def __init__(
        self,
        input_dim: int,
        action_dim: int,
        hidden_dims: list = [256, 256],
        num_superposition_states: int = 4,
    ):
        super(QuantumInspiredPolicyValue, self).__init__()

        self.num_superposition_states = num_superposition_states

        # Create multiple "quantum" pathways
        self.pathways = nn.ModuleList([
            nn.Sequential(
                nn.Linear(input_dim, hidden_dims[0]),
                nn.ReLU(),
                nn.Linear(hidden_dims[0], hidden_dims[1]),
                nn.ReLU(),
            )
            for _ in range(num_superposition_states)
        ])

        # Superposition weights (learnable)
        self.superposition_weights = nn.Parameter(
            torch.randn(num_superposition_states) / num_superposition_states
        )

        # Policy and value heads
        combined_dim = hidden_dims[1]
        self.policy_head = nn.Linear(combined_dim, action_dim)
        self.value_head = nn.Sequential(
            nn.Linear(combined_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Tanh(),
        )

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass with quantum-inspired superposition."""
        # Compute features through each pathway
        pathway_outputs = [pathway(x) for pathway in self.pathways]

        # Normalize superposition weights (quantum amplitude-like)
        weights = F.softmax(self.superposition_weights, dim=0)

        # Superpose the outputs (like quantum state collapse)
        combined = sum(w * out for w, out in zip(weights, pathway_outputs))

        # Policy and value
        policy_logits = self.policy_head(combined)
        value = self.value_head(combined)

        return policy_logits, value
