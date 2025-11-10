"""
Neural network models for BelRL.

Policy-Value networks for AlphaZero-style training.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple


class ResidualBlock(nn.Module):
    """Residual block for policy-value network."""

    def __init__(self, channels: int):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(channels)

    def forward(self, x):
        residual = x
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += residual
        out = F.relu(out)
        return out


class PolicyValueNetwork(nn.Module):
    """
    Combined policy-value network for board games.

    Architecture similar to AlphaZero:
    - Shared representation layers (ResNet-style)
    - Policy head (outputs action probabilities)
    - Value head (outputs state value)
    """

    def __init__(
        self,
        board_size: int,
        action_size: int,
        num_channels: int = 256,
        num_res_blocks: int = 19,
        input_channels: int = 17,  # e.g., for Go: 8 black + 8 white + 1 color
    ):
        """
        Args:
            board_size: Size of the board (e.g., 19 for Go, 8 for Chess)
            action_size: Number of possible actions
            num_channels: Number of channels in conv layers
            num_res_blocks: Number of residual blocks
            input_channels: Number of input feature planes
        """
        super().__init__()

        self.board_size = board_size
        self.action_size = action_size

        # Initial convolution
        self.conv_input = nn.Conv2d(input_channels, num_channels, kernel_size=3, padding=1)
        self.bn_input = nn.BatchNorm2d(num_channels)

        # Residual tower
        self.res_blocks = nn.ModuleList([
            ResidualBlock(num_channels) for _ in range(num_res_blocks)
        ])

        # Policy head
        self.policy_conv = nn.Conv2d(num_channels, 32, kernel_size=1)
        self.policy_bn = nn.BatchNorm2d(32)
        self.policy_fc = nn.Linear(32 * board_size * board_size, action_size)

        # Value head
        self.value_conv = nn.Conv2d(num_channels, 32, kernel_size=1)
        self.value_bn = nn.BatchNorm2d(32)
        self.value_fc1 = nn.Linear(32 * board_size * board_size, 256)
        self.value_fc2 = nn.Linear(256, 1)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass.

        Args:
            x: Input tensor of shape (batch, input_channels, board_size, board_size)

        Returns:
            policy: Action probabilities (batch, action_size)
            value: State value (batch, 1)
        """
        # Shared representation
        out = F.relu(self.bn_input(self.conv_input(x)))

        for block in self.res_blocks:
            out = block(out)

        # Policy head
        policy = F.relu(self.policy_bn(self.policy_conv(out)))
        policy = policy.view(policy.size(0), -1)
        policy = self.policy_fc(policy)
        policy = F.log_softmax(policy, dim=1)

        # Value head
        value = F.relu(self.value_bn(self.value_conv(out)))
        value = value.view(value.size(0), -1)
        value = F.relu(self.value_fc1(value))
        value = torch.tanh(self.value_fc2(value))

        return policy, value


class SimplePolicyValueNetwork(nn.Module):
    """
    Simplified policy-value network for smaller games or testing.

    Uses MLPs instead of CNNs.
    """

    def __init__(
        self,
        state_size: int,
        action_size: int,
        hidden_dim: int = 256,
        num_layers: int = 4,
    ):
        super().__init__()

        self.action_size = action_size

        # Shared layers
        layers = []
        prev_dim = state_size
        for _ in range(num_layers):
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.ReLU(),
                nn.BatchNorm1d(hidden_dim),
            ])
            prev_dim = hidden_dim

        self.shared = nn.Sequential(*layers)

        # Policy head
        self.policy_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, action_size),
        )

        # Value head
        self.value_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1),
            nn.Tanh(),
        )

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass.

        Args:
            x: Input tensor of shape (batch, state_size)

        Returns:
            policy: Action log-probabilities (batch, action_size)
            value: State value (batch, 1)
        """
        shared_out = self.shared(x)

        policy = self.policy_head(shared_out)
        policy = F.log_softmax(policy, dim=1)

        value = self.value_head(shared_out)

        return policy, value


class HexPolicyValueNetwork(nn.Module):
    """Policy-value network specifically for Hex game."""

    def __init__(self, board_size: int = 11, num_channels: int = 128):
        super().__init__()

        self.board_size = board_size
        action_size = board_size * board_size

        # Input: 2 channels (player 1 stones, player 2 stones)
        self.conv1 = nn.Conv2d(2, num_channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(num_channels)

        # Residual blocks
        self.res_blocks = nn.ModuleList([
            ResidualBlock(num_channels) for _ in range(6)
        ])

        # Policy head
        self.policy_conv = nn.Conv2d(num_channels, 16, kernel_size=1)
        self.policy_bn = nn.BatchNorm2d(16)
        self.policy_fc = nn.Linear(16 * board_size * board_size, action_size)

        # Value head
        self.value_conv = nn.Conv2d(num_channels, 16, kernel_size=1)
        self.value_bn = nn.BatchNorm2d(16)
        self.value_fc1 = nn.Linear(16 * board_size * board_size, 128)
        self.value_fc2 = nn.Linear(128, 1)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        out = F.relu(self.bn1(self.conv1(x)))

        for block in self.res_blocks:
            out = block(out)

        # Policy
        policy = F.relu(self.policy_bn(self.policy_conv(out)))
        policy = policy.view(policy.size(0), -1)
        policy = F.log_softmax(self.policy_fc(policy), dim=1)

        # Value
        value = F.relu(self.value_bn(self.value_conv(out)))
        value = value.view(value.size(0), -1)
        value = torch.tanh(self.value_fc2(F.relu(self.value_fc1(value))))

        return policy, value


if __name__ == '__main__':
    # Test networks
    batch_size = 4

    # Test PolicyValueNetwork (for Go/Chess)
    print("Testing PolicyValueNetwork...")
    net = PolicyValueNetwork(board_size=19, action_size=361, num_res_blocks=5)
    x = torch.randn(batch_size, 17, 19, 19)
    policy, value = net(x)
    print(f"  Input: {x.shape}")
    print(f"  Policy: {policy.shape}, Value: {value.shape}")

    # Test SimplePolicyValueNetwork
    print("\nTesting SimplePolicyValueNetwork...")
    simple_net = SimplePolicyValueNetwork(state_size=100, action_size=50)
    x = torch.randn(batch_size, 100)
    policy, value = simple_net(x)
    print(f"  Input: {x.shape}")
    print(f"  Policy: {policy.shape}, Value: {value.shape}")

    # Test HexPolicyValueNetwork
    print("\nTesting HexPolicyValueNetwork...")
    hex_net = HexPolicyValueNetwork(board_size=11)
    x = torch.randn(batch_size, 2, 11, 11)
    policy, value = hex_net(x)
    print(f"  Input: {x.shape}")
    print(f"  Policy: {policy.shape}, Value: {value.shape}")

    print("\n✅ All models tested successfully!")
