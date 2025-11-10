"""
RL agents using Belavkin filtering framework.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Tuple, Optional, Dict, Any, List
from collections import deque
import random

from belavkin_ml.rl.core import BelavkinFilter, QuantumState, LowRankBelavkinFilter


class BelavkinAgent:
    """
    Base agent using Belavkin filtering for belief state management.

    Args:
        state_dim (int): Dimension of (hidden) state space
        obs_dim (int): Dimension of observation space
        action_dim (int): Number of actions
        use_low_rank (bool): Use low-rank approximation
        n_particles (int): Number of particles for low-rank filter
        device (str): Device for computation
    """

    def __init__(
        self,
        state_dim: int,
        obs_dim: int,
        action_dim: int,
        use_low_rank: bool = False,
        n_particles: int = 100,
        device: str = 'cpu',
    ):
        self.state_dim = state_dim
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.device = device

        # Initialize Belavkin filter
        if use_low_rank:
            self.filter = LowRankBelavkinFilter(
                state_dim=state_dim,
                obs_dim=obs_dim,
                n_particles=n_particles,
                device=device,
            )
        else:
            self.filter = BelavkinFilter(
                state_dim=state_dim,
                obs_dim=obs_dim,
                device=device,
            )

    def select_action(self, observation: torch.Tensor) -> int:
        """
        Select action based on current belief state.

        To be implemented by subclasses.
        """
        raise NotImplementedError

    def update_belief(
        self,
        action: int,
        observation: torch.Tensor,
        reward: float,
    ) -> QuantumState:
        """
        Update belief state using Belavkin filtering.

        Args:
            action: Action taken
            observation: Observation received
            reward: Reward received

        Returns:
            updated_belief: New belief state
        """
        belief, innovation = self.filter.filter_step(
            action=action,
            observation=observation,
        )

        return belief

    def reset(self):
        """Reset agent state."""
        self.filter.reset()


class BelavkinDQN(BelavkinAgent):
    """
    DQN agent with Belavkin belief state representation.

    Uses belief state as input to Q-network instead of raw observations.

    Args:
        state_dim (int): State space dimension
        obs_dim (int): Observation dimension
        action_dim (int): Number of actions
        hidden_dims (list): Hidden layer dimensions for Q-network
        lr (float): Learning rate
        gamma (float): Discount factor
        epsilon (float): Exploration rate
        buffer_size (int): Replay buffer size
        batch_size (int): Batch size for training
        device (str): Device
    """

    def __init__(
        self,
        state_dim: int,
        obs_dim: int,
        action_dim: int,
        hidden_dims: List[int] = [128, 128],
        lr: float = 1e-3,
        gamma: float = 0.99,
        epsilon: float = 0.1,
        buffer_size: int = 10000,
        batch_size: int = 64,
        device: str = 'cpu',
        **kwargs
    ):
        super().__init__(state_dim, obs_dim, action_dim, device=device, **kwargs)

        self.gamma = gamma
        self.epsilon = epsilon
        self.batch_size = batch_size

        # Q-network (input: belief state distribution)
        self.q_network = self._build_q_network(state_dim, action_dim, hidden_dims).to(device)
        self.target_network = self._build_q_network(state_dim, action_dim, hidden_dims).to(device)
        self.target_network.load_state_dict(self.q_network.state_dict())

        self.optimizer = torch.optim.Adam(self.q_network.parameters(), lr=lr)

        # Replay buffer
        self.replay_buffer = deque(maxlen=buffer_size)

        # Current observation for belief update
        self.last_observation = None

    def _build_q_network(
        self,
        state_dim: int,
        action_dim: int,
        hidden_dims: List[int],
    ) -> nn.Module:
        """Build Q-network."""
        layers = []
        prev_dim = state_dim

        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.ReLU())
            prev_dim = hidden_dim

        layers.append(nn.Linear(prev_dim, action_dim))

        return nn.Sequential(*layers)

    def select_action(self, observation: torch.Tensor, training: bool = True) -> int:
        """
        Select action using epsilon-greedy policy.

        Args:
            observation: Current observation
            training: Whether in training mode (affects exploration)

        Returns:
            action: Selected action
        """
        # Update belief
        if self.last_observation is not None:
            # This should be called after update_belief
            pass

        # Get belief distribution
        belief_probs = self.filter.belief.get_probabilities()

        # Epsilon-greedy
        if training and random.random() < self.epsilon:
            return random.randint(0, self.action_dim - 1)

        # Greedy action
        with torch.no_grad():
            q_values = self.q_network(belief_probs.unsqueeze(0))
            action = q_values.argmax(dim=1).item()

        return action

    def store_transition(
        self,
        belief: torch.Tensor,
        action: int,
        reward: float,
        next_belief: torch.Tensor,
        done: bool,
    ):
        """Store transition in replay buffer."""
        self.replay_buffer.append((belief, action, reward, next_belief, done))

    def train_step(self) -> Optional[float]:
        """
        Perform one training step.

        Returns:
            loss: Training loss (None if buffer too small)
        """
        if len(self.replay_buffer) < self.batch_size:
            return None

        # Sample batch
        batch = random.sample(self.replay_buffer, self.batch_size)

        beliefs, actions, rewards, next_beliefs, dones = zip(*batch)

        beliefs = torch.stack(beliefs).to(self.device)
        actions = torch.tensor(actions, dtype=torch.long, device=self.device)
        rewards = torch.tensor(rewards, dtype=torch.float32, device=self.device)
        next_beliefs = torch.stack(next_beliefs).to(self.device)
        dones = torch.tensor(dones, dtype=torch.float32, device=self.device)

        # Compute Q-values
        q_values = self.q_network(beliefs)
        q_values = q_values.gather(1, actions.unsqueeze(1)).squeeze(1)

        # Compute target Q-values
        with torch.no_grad():
            next_q_values = self.target_network(next_beliefs).max(dim=1)[0]
            target_q_values = rewards + self.gamma * next_q_values * (1 - dones)

        # Compute loss
        loss = F.mse_loss(q_values, target_q_values)

        # Optimize
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return loss.item()

    def update_target_network(self):
        """Update target network."""
        self.target_network.load_state_dict(self.q_network.state_dict())


class BelavkinPPO(BelavkinAgent):
    """
    PPO agent with Belavkin belief state representation.

    Args:
        state_dim (int): State space dimension
        obs_dim (int): Observation dimension
        action_dim (int): Number of actions
        hidden_dims (list): Hidden layer dimensions
        lr (float): Learning rate
        gamma (float): Discount factor
        epsilon (float): PPO clipping parameter
        device (str): Device
    """

    def __init__(
        self,
        state_dim: int,
        obs_dim: int,
        action_dim: int,
        hidden_dims: List[int] = [128, 128],
        lr: float = 3e-4,
        gamma: float = 0.99,
        epsilon: float = 0.2,
        device: str = 'cpu',
        **kwargs
    ):
        super().__init__(state_dim, obs_dim, action_dim, device=device, **kwargs)

        self.gamma = gamma
        self.epsilon = epsilon

        # Policy network
        self.policy_network = self._build_network(
            state_dim, action_dim, hidden_dims, output_activation='softmax'
        ).to(device)

        # Value network
        self.value_network = self._build_network(
            state_dim, 1, hidden_dims, output_activation='linear'
        ).to(device)

        self.optimizer = torch.optim.Adam(
            list(self.policy_network.parameters()) +
            list(self.value_network.parameters()),
            lr=lr
        )

        # Episode buffer
        self.episode_buffer = []

    def _build_network(
        self,
        input_dim: int,
        output_dim: int,
        hidden_dims: List[int],
        output_activation: str,
    ) -> nn.Module:
        """Build neural network."""
        layers = []
        prev_dim = input_dim

        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.ReLU())
            prev_dim = hidden_dim

        layers.append(nn.Linear(prev_dim, output_dim))

        if output_activation == 'softmax':
            layers.append(nn.Softmax(dim=-1))

        return nn.Sequential(*layers)

    def select_action(self, observation: torch.Tensor, training: bool = True) -> Tuple[int, float]:
        """
        Select action using policy network.

        Returns:
            action, log_prob
        """
        # Get belief distribution
        belief_probs = self.filter.belief.get_probabilities()

        # Get action probabilities
        with torch.no_grad() if not training else torch.enable_grad():
            action_probs = self.policy_network(belief_probs.unsqueeze(0)).squeeze(0)

        # Sample action
        if training:
            action_dist = torch.distributions.Categorical(action_probs)
            action = action_dist.sample()
            log_prob = action_dist.log_prob(action)
            return action.item(), log_prob.item()
        else:
            return action_probs.argmax().item(), 0.0

    def store_transition(
        self,
        belief: torch.Tensor,
        action: int,
        reward: float,
        log_prob: float,
        value: float,
    ):
        """Store transition in episode buffer."""
        self.episode_buffer.append({
            'belief': belief,
            'action': action,
            'reward': reward,
            'log_prob': log_prob,
            'value': value,
        })

    def train_episode(self) -> Tuple[float, float]:
        """
        Train on collected episode.

        Returns:
            policy_loss, value_loss
        """
        if len(self.episode_buffer) == 0:
            return 0.0, 0.0

        # Compute returns and advantages
        returns = []
        advantages = []
        G = 0

        for t in reversed(range(len(self.episode_buffer))):
            G = self.episode_buffer[t]['reward'] + self.gamma * G
            returns.insert(0, G)

        returns = torch.tensor(returns, dtype=torch.float32, device=self.device)

        # Normalize returns
        returns = (returns - returns.mean()) / (returns.std() + 1e-8)

        # Extract data
        beliefs = torch.stack([t['belief'] for t in self.episode_buffer]).to(self.device)
        actions = torch.tensor([t['action'] for t in self.episode_buffer],
                                dtype=torch.long, device=self.device)
        old_log_probs = torch.tensor([t['log_prob'] for t in self.episode_buffer],
                                      dtype=torch.float32, device=self.device)

        # Compute current values
        values = self.value_network(beliefs).squeeze(1)
        advantages = returns - values.detach()

        # Compute current policy log probs
        action_probs = self.policy_network(beliefs)
        action_dist = torch.distributions.Categorical(action_probs)
        new_log_probs = action_dist.log_prob(actions)

        # PPO loss
        ratio = torch.exp(new_log_probs - old_log_probs)
        surr1 = ratio * advantages
        surr2 = torch.clamp(ratio, 1 - self.epsilon, 1 + self.epsilon) * advantages
        policy_loss = -torch.min(surr1, surr2).mean()

        # Value loss
        value_loss = F.mse_loss(values, returns)

        # Total loss
        loss = policy_loss + 0.5 * value_loss

        # Optimize
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # Clear buffer
        self.episode_buffer = []

        return policy_loss.item(), value_loss.item()
