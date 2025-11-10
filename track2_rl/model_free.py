"""
Model-Free Belavkin RL

Approach: Directly learn Q-function or policy without explicit model.
Adapt Belavkin filtering for value function learning.

Key Innovation: Treat Q-function as "observable" in quantum framework.
Learning becomes a filtering problem over value space.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple
import numpy as np

from .belavkin_rl import BelavkinRLAgent, QuantumBeliefNetwork


class QNetwork(nn.Module):
    """
    Q-network with uncertainty estimation.

    Outputs both Q-value and epistemic uncertainty (related to quantum measurement).
    """

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        hidden_dim: int = 256,
        num_layers: int = 3,
    ):
        super().__init__()

        layers = []
        input_dim = state_dim + action_dim

        for i in range(num_layers):
            layers.append(nn.Linear(input_dim, hidden_dim))
            layers.append(nn.ReLU())
            input_dim = hidden_dim

        self.shared = nn.Sequential(*layers)

        # Q-value head
        self.q_head = nn.Linear(hidden_dim, 1)

        # Uncertainty head (epistemic uncertainty)
        self.log_uncertainty_head = nn.Linear(hidden_dim, 1)

    def forward(
        self,
        state: torch.Tensor,
        action: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute Q-value and uncertainty.

        Args:
            state: State tensor
            action: Action tensor

        Returns:
            Tuple of (q_value, log_uncertainty)
        """
        x = torch.cat([state, action], dim=-1)
        h = self.shared(x)

        q_value = self.q_head(h).squeeze(-1)
        log_uncertainty = self.log_uncertainty_head(h).squeeze(-1)
        log_uncertainty = torch.clamp(log_uncertainty, -10, 2)

        return q_value, log_uncertainty


class ModelFreeBelavkinRL(BelavkinRLAgent):
    """
    Model-free Belavkin RL agent.

    Uses quantum filtering principles for value function learning without
    explicitly modeling environment dynamics.
    """

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        belief_dim: int = 64,
        hidden_dim: int = 256,
        lr: float = 3e-4,
        gamma: float = 0.99,
        tau: float = 0.005,  # Soft update rate
        measurement_strength: float = 0.1,  # Quantum coupling
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
    ):
        super().__init__(state_dim, action_dim, belief_dim, device)

        self.lr = lr
        self.gamma = gamma
        self.tau = tau
        self.measurement_strength = measurement_strength

        # Belief network for state encoding
        self.belief_network = QuantumBeliefNetwork(
            obs_dim=state_dim,
            belief_dim=belief_dim,
            hidden_dim=hidden_dim,
        ).to(device)

        # Q-networks (twin networks for stability)
        self.q1 = QNetwork(state_dim, action_dim, hidden_dim).to(device)
        self.q2 = QNetwork(state_dim, action_dim, hidden_dim).to(device)

        # Target Q-networks
        self.q1_target = QNetwork(state_dim, action_dim, hidden_dim).to(device)
        self.q2_target = QNetwork(state_dim, action_dim, hidden_dim).to(device)

        # Initialize targets
        self.q1_target.load_state_dict(self.q1.state_dict())
        self.q2_target.load_state_dict(self.q2.state_dict())

        # Policy network
        self.policy = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim),
            nn.Tanh(),
        ).to(device)

        # Optimizers
        self.q1_optimizer = torch.optim.Adam(self.q1.parameters(), lr=lr)
        self.q2_optimizer = torch.optim.Adam(self.q2.parameters(), lr=lr)
        self.policy_optimizer = torch.optim.Adam(self.policy.parameters(), lr=lr)
        self.belief_optimizer = torch.optim.Adam(self.belief_network.parameters(), lr=lr)

        # Exploration parameters
        self.action_noise_std = 0.1

    def select_action(
        self,
        observation: torch.Tensor,
        training: bool = True,
    ) -> torch.Tensor:
        """
        Select action using policy with optional exploration noise.

        Args:
            observation: Current observation
            training: Whether in training mode

        Returns:
            Selected action
        """
        with torch.no_grad():
            action = self.policy(observation)

            if training:
                # Add exploration noise (quantum-inspired stochasticity)
                # Noise strength can be modulated by belief uncertainty
                belief_mean, belief_log_std = self.belief_network(observation)
                uncertainty = torch.exp(belief_log_std).mean()

                # Adaptive noise based on uncertainty
                noise_scale = self.action_noise_std * (1.0 + uncertainty.item())
                noise = torch.randn_like(action) * noise_scale
                action = action + noise
                action = torch.clamp(action, -1.0, 1.0)

        return action

    def update_belief(
        self,
        observation: torch.Tensor,
        action: torch.Tensor,
        reward: float,
        next_observation: torch.Tensor,
        done: bool,
    ):
        """
        Update belief state (handled implicitly through training).

        Args:
            observation: Current observation
            action: Action taken
            reward: Reward received
            next_observation: Next observation
            done: Episode done flag
        """
        # In model-free setting, belief updates happen through Q-learning
        # The belief network learns to represent uncertainty implicitly
        pass

    def train_step(
        self,
        batch: Dict[str, torch.Tensor],
    ) -> Dict[str, float]:
        """
        Perform one training step (similar to SAC but with quantum-inspired modifications).

        Args:
            batch: Experience batch

        Returns:
            Training metrics
        """
        states = batch['state'].to(self.device)
        actions = batch['action'].to(self.device)
        rewards = batch['reward'].to(self.device)
        next_states = batch['next_state'].to(self.device)
        dones = batch['done'].to(self.device)

        metrics = {}

        # 1. Train Q-networks with Belavkin-inspired updates
        q_loss = self._train_q_networks(states, actions, rewards, next_states, dones)
        metrics['q_loss'] = q_loss

        # 2. Train policy
        policy_loss = self._train_policy(states)
        metrics['policy_loss'] = policy_loss

        # 3. Train belief network
        belief_loss = self._train_belief(states, next_states)
        metrics['belief_loss'] = belief_loss

        # 4. Soft update target networks
        self._soft_update_targets()

        return metrics

    def _train_q_networks(
        self,
        states: torch.Tensor,
        actions: torch.Tensor,
        rewards: torch.Tensor,
        next_states: torch.Tensor,
        dones: torch.Tensor,
    ) -> float:
        """
        Train Q-networks with quantum-inspired modifications.

        The Belavkin twist: Use uncertainty estimates to modulate learning.
        """
        # Compute current Q-values and uncertainties
        q1_values, q1_log_uncertainty = self.q1(states, actions)
        q2_values, q2_log_uncertainty = self.q2(states, actions)

        # Compute target Q-values
        with torch.no_grad():
            # Sample next actions from policy
            next_actions = self.policy(next_states)

            # Compute target Q-values (take minimum for conservatism)
            q1_next, q1_next_unc = self.q1_target(next_states, next_actions)
            q2_next, q2_next_unc = self.q2_target(next_states, next_actions)
            q_next = torch.min(q1_next, q2_next)

            # Belavkin modification: incorporate uncertainty into targets
            # Higher uncertainty → more conservative estimates
            uncertainty_penalty = self.measurement_strength * torch.exp(
                0.5 * (q1_next_unc + q2_next_unc)
            )
            q_next = q_next - uncertainty_penalty

            # TD target
            q_target = rewards + self.gamma * q_next * (1 - dones.float())

        # Compute losses
        # Standard TD loss
        q1_loss_td = F.mse_loss(q1_values, q_target)
        q2_loss_td = F.mse_loss(q2_values, q_target)

        # Uncertainty regularization (encourage calibrated uncertainty)
        # Want: high uncertainty when predictions are unreliable
        td_error = torch.abs(q1_values.detach() - q_target)
        uncertainty_loss = F.mse_loss(
            torch.exp(q1_log_uncertainty),
            td_error.detach()
        )

        # Total losses
        q1_loss = q1_loss_td + 0.1 * uncertainty_loss
        q2_loss = q2_loss_td + 0.1 * uncertainty_loss

        # Optimize Q1
        self.q1_optimizer.zero_grad()
        q1_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.q1.parameters(), 1.0)
        self.q1_optimizer.step()

        # Optimize Q2
        self.q2_optimizer.zero_grad()
        q2_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.q2.parameters(), 1.0)
        self.q2_optimizer.step()

        return (q1_loss.item() + q2_loss.item()) / 2

    def _train_policy(
        self,
        states: torch.Tensor,
    ) -> float:
        """Train policy to maximize Q-values."""
        # Sample actions from current policy
        policy_actions = self.policy(states)

        # Compute Q-values
        q1_values, _ = self.q1(states, policy_actions)
        q2_values, _ = self.q2(states, policy_actions)
        q_values = torch.min(q1_values, q2_values)

        # Policy loss: maximize Q-values
        policy_loss = -q_values.mean()

        # Action regularization
        action_reg = 0.01 * (policy_actions ** 2).mean()
        policy_loss = policy_loss + action_reg

        # Optimize
        self.policy_optimizer.zero_grad()
        policy_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy.parameters(), 1.0)
        self.policy_optimizer.step()

        return policy_loss.item()

    def _train_belief(
        self,
        states: torch.Tensor,
        next_states: torch.Tensor,
    ) -> float:
        """
        Train belief network to capture state uncertainty.

        This is the quantum filtering component: maintain belief over states.
        """
        # Encode states to belief distributions
        belief_mean, belief_log_std = self.belief_network(states)
        next_belief_mean, next_belief_log_std = self.belief_network(next_states)

        # Temporal consistency: beliefs should vary smoothly
        consistency_loss = F.mse_loss(belief_mean, next_belief_mean.detach())

        # KL regularization (prevent collapse)
        belief_std = torch.exp(belief_log_std)
        kl_loss = 0.5 * (belief_mean**2 + belief_std**2 - 2*belief_log_std - 1).mean()

        # Total belief loss
        belief_loss = consistency_loss + 0.01 * kl_loss

        # Optimize
        self.belief_optimizer.zero_grad()
        belief_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.belief_network.parameters(), 1.0)
        self.belief_optimizer.step()

        return belief_loss.item()

    def _soft_update_targets(self):
        """Soft update target networks."""
        for param, target_param in zip(self.q1.parameters(), self.q1_target.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

        for param, target_param in zip(self.q2.parameters(), self.q2_target.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)


def test_model_free():
    """Quick test of model-free Belavkin RL."""
    print("Testing Model-Free Belavkin RL...")

    state_dim = 4
    action_dim = 2
    batch_size = 32

    agent = ModelFreeBelavkinRL(
        state_dim=state_dim,
        action_dim=action_dim,
        belief_dim=16,
        hidden_dim=64,
    )

    # Test action selection
    state = torch.randn(state_dim)
    action = agent.select_action(state)
    print(f"  State shape: {state.shape}")
    print(f"  Action shape: {action.shape}")

    # Test training step
    batch = {
        'state': torch.randn(batch_size, state_dim),
        'action': torch.randn(batch_size, action_dim),
        'reward': torch.randn(batch_size),
        'next_state': torch.randn(batch_size, state_dim),
        'done': torch.randint(0, 2, (batch_size,)),
    }

    metrics = agent.train_step(batch)
    print(f"  Training metrics: {metrics}")

    print("✓ Model-free Belavkin RL test passed!\n")


if __name__ == "__main__":
    test_model_free()
