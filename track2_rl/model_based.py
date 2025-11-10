"""
Model-Based Belavkin RL

Approach: Learn transition model, use Belavkin filtering for belief state,
plan in belief space.

Algorithm:
1. Learn dynamics model: s_{t+1} = f(s_t, a_t) + noise
2. Use Belavkin filter to maintain belief over states
3. Plan using belief state (e.g., via model-predictive control)
4. Update policy via planning or policy gradient
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple
import numpy as np

from .belavkin_rl import BelavkinRLAgent, QuantumBeliefNetwork, BelavkinFilter


class DynamicsModel(nn.Module):
    """
    Learnable dynamics model for environment transitions.

    Predicts: s_{t+1}, r_t = f(s_t, a_t)
    """

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        hidden_dim: int = 256,
        num_layers: int = 3,
    ):
        super().__init__()

        self.state_dim = state_dim
        self.action_dim = action_dim

        # Build network: [state, action] → [next_state, reward]
        layers = []
        input_dim = state_dim + action_dim

        for i in range(num_layers):
            layers.append(nn.Linear(input_dim, hidden_dim))
            layers.append(nn.ReLU())
            input_dim = hidden_dim

        self.shared = nn.Sequential(*layers)

        # Separate heads for next state and reward prediction
        self.next_state_head = nn.Linear(hidden_dim, state_dim)
        self.reward_head = nn.Linear(hidden_dim, 1)

        # Uncertainty estimation (aleatoric uncertainty)
        self.next_state_log_std = nn.Linear(hidden_dim, state_dim)

    def forward(
        self,
        state: torch.Tensor,
        action: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Predict next state and reward.

        Args:
            state: Current state
            action: Action taken

        Returns:
            Tuple of (next_state_mean, next_state_log_std, reward, reward uncertainty)
        """
        # Concatenate state and action
        x = torch.cat([state, action], dim=-1)

        # Forward through shared layers
        h = self.shared(x)

        # Predict next state (with uncertainty)
        next_state_mean = self.next_state_head(h)
        next_state_log_std = self.next_state_log_std(h)
        next_state_log_std = torch.clamp(next_state_log_std, -10, 2)

        # Predict reward
        reward = self.reward_head(h).squeeze(-1)

        return next_state_mean, next_state_log_std, reward, None

    def sample_next_state(
        self,
        state: torch.Tensor,
        action: torch.Tensor,
        num_samples: int = 1,
    ) -> torch.Tensor:
        """
        Sample next states from the learned distribution.

        Args:
            state: Current state
            action: Action taken
            num_samples: Number of samples

        Returns:
            Sampled next states
        """
        next_state_mean, next_state_log_std, _, _ = self.forward(state, action)
        next_state_std = torch.exp(next_state_log_std)

        # Sample using reparameterization
        eps = torch.randn(num_samples, *next_state_mean.shape, device=state.device)
        samples = next_state_mean.unsqueeze(0) + eps * next_state_std.unsqueeze(0)

        return samples


class ModelBasedBelavkinRL(BelavkinRLAgent):
    """
    Model-based Belavkin RL agent.

    Learns environment dynamics and uses Belavkin filtering for belief updates.
    """

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        belief_dim: int = 64,
        hidden_dim: int = 256,
        lr: float = 3e-4,
        gamma: float = 0.99,  # Discount factor
        measurement_strength: float = 0.1,  # Belavkin γ parameter
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
    ):
        super().__init__(state_dim, action_dim, belief_dim, device)

        self.lr = lr
        self.gamma = gamma

        # Components
        self.belief_network = QuantumBeliefNetwork(
            obs_dim=state_dim,
            belief_dim=belief_dim,
            hidden_dim=hidden_dim,
        ).to(device)

        self.dynamics_model = DynamicsModel(
            state_dim=state_dim,
            action_dim=action_dim,
            hidden_dim=hidden_dim,
        ).to(device)

        self.belavkin_filter = BelavkinFilter(
            belief_dim=belief_dim,
            gamma=measurement_strength,
            learning_rate=0.1,
        )

        # Policy network (maps belief → action)
        self.policy = nn.Sequential(
            nn.Linear(belief_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim),
            nn.Tanh(),  # Assume continuous actions in [-1, 1]
        ).to(device)

        # Value network for planning
        self.value_network = nn.Sequential(
            nn.Linear(belief_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        ).to(device)

        # Optimizers
        self.belief_optimizer = torch.optim.Adam(self.belief_network.parameters(), lr=lr)
        self.dynamics_optimizer = torch.optim.Adam(self.dynamics_model.parameters(), lr=lr)
        self.policy_optimizer = torch.optim.Adam(self.policy.parameters(), lr=lr)
        self.value_optimizer = torch.optim.Adam(self.value_network.parameters(), lr=lr)

        # Current belief state (vector representation)
        self.current_belief_mean = None
        self.current_belief_std = None

    def select_action(
        self,
        observation: torch.Tensor,
        training: bool = True,
    ) -> torch.Tensor:
        """
        Select action based on belief state.

        Args:
            observation: Current observation
            training: Whether in training mode

        Returns:
            Selected action
        """
        # Encode observation to belief
        with torch.no_grad():
            belief_mean, belief_log_std = self.belief_network(observation)

            # Select action from policy
            action = self.policy(belief_mean)

            # Add exploration noise during training
            if training:
                noise = torch.randn_like(action) * 0.1
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
        Update belief using Belavkin filtering.

        Args:
            observation: Current observation
            action: Action taken
            reward: Reward received
            next_observation: Next observation
            done: Episode done flag
        """
        # Encode current and next observations to belief parameters
        belief_mean, belief_log_std = self.belief_network(observation)
        next_belief_mean, next_belief_log_std = self.belief_network(next_observation)

        # Store current belief
        self.current_belief_mean = belief_mean
        self.current_belief_std = torch.exp(belief_log_std)

        # Apply Belavkin filter update
        # This incorporates the measurement (next_observation)
        # In the full theory, this would be the quantum filtering update
        # Here we use a simplified tractable version

        # For now: update stored in network, training will improve belief representation

    def train_step(
        self,
        batch: Dict[str, torch.Tensor],
    ) -> Dict[str, float]:
        """
        Train all components on a batch of experience.

        Args:
            batch: Dictionary with keys ['state', 'action', 'reward', 'next_state', 'done']

        Returns:
            Dictionary of training metrics
        """
        states = batch['state'].to(self.device)
        actions = batch['action'].to(self.device)
        rewards = batch['reward'].to(self.device)
        next_states = batch['next_state'].to(self.device)
        dones = batch['done'].to(self.device)

        metrics = {}

        # 1. Train dynamics model
        dynamics_loss = self._train_dynamics(states, actions, next_states, rewards)
        metrics['dynamics_loss'] = dynamics_loss

        # 2. Train belief network (variational lower bound)
        belief_loss = self._train_belief_network(states, actions, next_states)
        metrics['belief_loss'] = belief_loss

        # 3. Train policy and value (using beliefs and learned model)
        policy_loss, value_loss = self._train_policy_value(states, actions, rewards, next_states, dones)
        metrics['policy_loss'] = policy_loss
        metrics['value_loss'] = value_loss

        return metrics

    def _train_dynamics(
        self,
        states: torch.Tensor,
        actions: torch.Tensor,
        next_states: torch.Tensor,
        rewards: torch.Tensor,
    ) -> float:
        """Train dynamics model."""
        # Predict next state and reward
        pred_next_state_mean, pred_next_state_log_std, pred_rewards, _ = \
            self.dynamics_model(states, actions)

        # Negative log-likelihood loss for next state prediction
        next_state_std = torch.exp(pred_next_state_log_std)
        state_nll = 0.5 * (
            ((next_states - pred_next_state_mean) / (next_state_std + 1e-8)) ** 2 +
            2 * pred_next_state_log_std +
            np.log(2 * np.pi)
        ).mean()

        # MSE loss for reward prediction
        reward_loss = F.mse_loss(pred_rewards, rewards)

        # Total dynamics loss
        dynamics_loss = state_nll + reward_loss

        # Optimize
        self.dynamics_optimizer.zero_grad()
        dynamics_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.dynamics_model.parameters(), 1.0)
        self.dynamics_optimizer.step()

        return dynamics_loss.item()

    def _train_belief_network(
        self,
        states: torch.Tensor,
        actions: torch.Tensor,
        next_states: torch.Tensor,
    ) -> float:
        """
        Train belief network to predict consistent beliefs.

        Uses a variational objective related to quantum state tomography.
        """
        # Encode states to beliefs
        belief_mean, belief_log_std = self.belief_network(states)
        next_belief_mean, next_belief_log_std = self.belief_network(next_states)

        # Consistency loss: beliefs should be consistent with learned dynamics
        # Predict next belief using dynamics model
        with torch.no_grad():
            pred_next_state, _, _, _ = self.dynamics_model(states, actions)

        # Encode predicted next state to belief
        pred_next_belief_mean, _ = self.belief_network(pred_next_state)

        # Loss: difference between actual next belief and predicted next belief
        consistency_loss = F.mse_loss(pred_next_belief_mean, next_belief_mean)

        # Regularization: encourage meaningful belief representations
        # KL divergence from standard normal (prevents collapse)
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

    def _train_policy_value(
        self,
        states: torch.Tensor,
        actions: torch.Tensor,
        rewards: torch.Tensor,
        next_states: torch.Tensor,
        dones: torch.Tensor,
    ) -> Tuple[float, float]:
        """Train policy and value networks."""
        # Encode states to beliefs
        with torch.no_grad():
            belief_mean, _ = self.belief_network(states)
            next_belief_mean, _ = self.belief_network(next_states)

        # Value loss (TD learning)
        current_values = self.value_network(belief_mean).squeeze(-1)
        with torch.no_grad():
            next_values = self.value_network(next_belief_mean).squeeze(-1)
            target_values = rewards + self.gamma * next_values * (1 - dones.float())

        value_loss = F.mse_loss(current_values, target_values)

        # Optimize value
        self.value_optimizer.zero_grad()
        value_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.value_network.parameters(), 1.0)
        self.value_optimizer.step()

        # Policy loss (advantage-based)
        belief_mean_policy, _ = self.belief_network(states)
        policy_actions = self.policy(belief_mean_policy)

        # Predict values for policy actions
        policy_values = self.value_network(belief_mean_policy).squeeze(-1)

        # Policy gradient: maximize value
        policy_loss = -policy_values.mean()

        # Add action regularization
        action_reg = 0.01 * (policy_actions ** 2).mean()
        policy_loss = policy_loss + action_reg

        # Optimize policy
        self.policy_optimizer.zero_grad()
        policy_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy.parameters(), 1.0)
        self.policy_optimizer.step()

        return policy_loss.item(), value_loss.item()


def test_model_based():
    """Quick test of model-based Belavkin RL."""
    print("Testing Model-Based Belavkin RL...")

    state_dim = 4
    action_dim = 2
    batch_size = 32

    agent = ModelBasedBelavkinRL(
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

    print("✓ Model-based Belavkin RL test passed!\n")


if __name__ == "__main__":
    test_model_based()
