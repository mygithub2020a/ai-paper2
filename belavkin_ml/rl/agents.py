"""
Belavkin RL Agents

Implements RL algorithms using Belavkin filtering for belief state management.

Agents:
1. ModelBasedBelavkinAgent: Learns transition model, uses Belavkin filtering
2. ModelFreeBelavkinAgent: Direct Q-learning with quantum-inspired updates
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Tuple, Optional, Dict, Any
from collections import deque
import random

from belavkin_ml.rl.belief_state import BelavkinBeliefState, BeliefStateConfig


class TransitionModel(nn.Module):
    """
    Neural network model for state transitions.

    Predicts: next_state ~ P(s' | s, a)
    """

    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim + action_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, state_dim),
        )

    def forward(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        x = torch.cat([state, action], dim=-1)
        return self.net(x)


class RewardModel(nn.Module):
    """Neural network model for reward prediction."""

    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim + action_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        x = torch.cat([state, action], dim=-1)
        return self.net(x)


class PolicyNetwork(nn.Module):
    """Policy network for action selection."""

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        hidden_dim: int = 128,
        discrete: bool = True,
    ):
        super().__init__()
        self.discrete = discrete
        self.action_dim = action_dim

        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )

        if discrete:
            self.action_head = nn.Linear(hidden_dim, action_dim)
        else:
            self.mean_head = nn.Linear(hidden_dim, action_dim)
            self.log_std_head = nn.Linear(hidden_dim, action_dim)

    def forward(
        self, state: torch.Tensor
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        features = self.net(state)

        if self.discrete:
            logits = self.action_head(features)
            return logits, None
        else:
            mean = self.mean_head(features)
            log_std = self.log_std_head(features)
            return mean, log_std


class ValueNetwork(nn.Module):
    """Value function approximator."""

    def __init__(self, state_dim: int, hidden_dim: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        return self.net(state).squeeze(-1)


class ModelBasedBelavkinAgent:
    """
    Model-based RL agent using Belavkin filtering.

    Algorithm:
    1. Learn transition and reward models
    2. Maintain belief state using Belavkin filtering
    3. Plan in belief space using learned models
    4. Update policy via policy gradient

    Args:
        state_dim: State space dimension
        action_dim: Action space dimension
        observation_dim: Observation space dimension
        belief_config: Configuration for belief state
        lr: Learning rate
        gamma: Discount factor
        device: torch device
    """

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        observation_dim: int,
        belief_config: Optional[BeliefStateConfig] = None,
        lr: float = 3e-4,
        gamma: float = 0.99,
        device: str = "cpu",
        discrete_actions: bool = True,
    ):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.observation_dim = observation_dim
        self.gamma = gamma
        self.device = device
        self.discrete_actions = discrete_actions

        # Create belief state
        if belief_config is None:
            belief_config = BeliefStateConfig(
                state_dim=state_dim,
                action_dim=action_dim,
                observation_dim=observation_dim,
            )
        self.belief_state = BelavkinBeliefState(belief_config, device)

        # Create models
        self.transition_model = TransitionModel(state_dim, action_dim).to(device)
        self.reward_model = RewardModel(state_dim, action_dim).to(device)
        self.policy = PolicyNetwork(
            state_dim, action_dim, discrete=discrete_actions
        ).to(device)
        self.value = ValueNetwork(state_dim).to(device)

        # Optimizers
        self.model_optimizer = torch.optim.Adam(
            list(self.transition_model.parameters())
            + list(self.reward_model.parameters()),
            lr=lr,
        )
        self.policy_optimizer = torch.optim.Adam(self.policy.parameters(), lr=lr)
        self.value_optimizer = torch.optim.Adam(self.value.parameters(), lr=lr)

        # Replay buffer
        self.buffer = deque(maxlen=100000)

    def select_action(
        self, observation: torch.Tensor, explore: bool = True
    ) -> torch.Tensor:
        """
        Select action based on current belief state.

        Args:
            observation: Current observation
            explore: Whether to add exploration noise

        Returns:
            action: Selected action
        """
        # Get mean belief state
        belief_state = self.belief_state.get_mean_state()

        with torch.no_grad():
            if self.discrete_actions:
                logits, _ = self.policy(belief_state.unsqueeze(0))
                probs = F.softmax(logits, dim=-1)

                if explore:
                    # Add exploration based on belief uncertainty
                    uncertainty = self.belief_state.get_uncertainty()
                    epsilon = min(0.5, uncertainty * 0.1)
                    if random.random() < epsilon:
                        action = torch.randint(0, self.action_dim, (1,))
                    else:
                        action = torch.multinomial(probs, 1)
                else:
                    action = torch.argmax(probs, dim=-1)

                return action.squeeze()
            else:
                mean, log_std = self.policy(belief_state.unsqueeze(0))

                if explore:
                    std = torch.exp(log_std)
                    action = mean + std * torch.randn_like(mean)
                else:
                    action = mean

                return action.squeeze()

    def update_belief(
        self, action: torch.Tensor, observation: torch.Tensor, reward: float
    ):
        """Update belief state with new experience."""
        # Prediction step
        self.belief_state.predict(action, dt=1.0)

        # Update step
        self.belief_state.update(observation, reward)

    def store_transition(
        self,
        state: torch.Tensor,
        action: torch.Tensor,
        reward: float,
        next_state: torch.Tensor,
        done: bool,
    ):
        """Store transition in replay buffer."""
        self.buffer.append((state, action, reward, next_state, done))

    def train_models(self, batch_size: int = 64, n_epochs: int = 1):
        """Train transition and reward models."""
        if len(self.buffer) < batch_size:
            return {}

        losses = {"transition_loss": 0.0, "reward_loss": 0.0}

        for _ in range(n_epochs):
            # Sample batch
            batch = random.sample(self.buffer, batch_size)
            states, actions, rewards, next_states, dones = zip(*batch)

            states = torch.stack([s.to(self.device) for s in states])
            actions = torch.stack([a.to(self.device) for a in actions])
            rewards = torch.tensor(rewards, dtype=torch.float32, device=self.device)
            next_states = torch.stack([s.to(self.device) for s in next_states])

            # One-hot encode actions if discrete
            if self.discrete_actions:
                actions_onehot = F.one_hot(
                    actions.long(), num_classes=self.action_dim
                ).float()
            else:
                actions_onehot = actions

            # Train transition model
            pred_next_states = self.transition_model(states, actions_onehot)
            transition_loss = F.mse_loss(pred_next_states, next_states)

            # Train reward model
            pred_rewards = self.reward_model(states, actions_onehot).squeeze()
            reward_loss = F.mse_loss(pred_rewards, rewards)

            # Combined loss
            model_loss = transition_loss + reward_loss

            self.model_optimizer.zero_grad()
            model_loss.backward()
            self.model_optimizer.step()

            losses["transition_loss"] += transition_loss.item()
            losses["reward_loss"] += reward_loss.item()

        losses = {k: v / n_epochs for k, v in losses.items()}
        return losses

    def train_policy(self, batch_size: int = 64):
        """Train policy using model-based rollouts."""
        if len(self.buffer) < batch_size:
            return {}

        # Sample initial states
        batch = random.sample(self.buffer, batch_size)
        states = torch.stack([s[0].to(self.device) for s in batch])

        # Perform short rollouts in imagination
        total_return = 0
        current_states = states

        for step in range(10):  # 10-step lookahead
            # Get actions from policy
            if self.discrete_actions:
                logits, _ = self.policy(current_states)
                probs = F.softmax(logits, dim=-1)
                actions = torch.multinomial(probs, 1).squeeze(-1)
                actions_onehot = F.one_hot(
                    actions, num_classes=self.action_dim
                ).float()
            else:
                mean, log_std = self.policy(current_states)
                std = torch.exp(log_std)
                actions = mean + std * torch.randn_like(mean)
                actions_onehot = actions

            # Predict next states and rewards
            next_states = self.transition_model(current_states, actions_onehot)
            rewards = self.reward_model(current_states, actions_onehot).squeeze()

            # Accumulate discounted return
            total_return = total_return + (self.gamma ** step) * rewards

            current_states = next_states

        # Policy gradient: maximize expected return
        policy_loss = -total_return.mean()

        self.policy_optimizer.zero_grad()
        policy_loss.backward()
        self.policy_optimizer.step()

        return {"policy_loss": policy_loss.item()}

    def reset(self):
        """Reset belief state for new episode."""
        self.belief_state.reset()


class ModelFreeBelavkinAgent:
    """
    Model-free Q-learning agent with Belavkin-inspired updates.

    Instead of learning transition models, directly learns Q-function with
    quantum-inspired update rules similar to the Belavkin optimizer.

    Update rule for Q-values:
        Q(s,a) ← Q(s,a) - [γ*(∇Q)² + η*∇Q]Δt + β*∇Q*ε

    Args:
        state_dim: State space dimension
        action_dim: Action space dimension
        lr: Learning rate
        gamma: Discount factor (RL)
        damping: Damping factor γ (Belavkin)
        exploration: Exploration factor β (Belavkin)
        device: torch device
    """

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        lr: float = 3e-4,
        gamma: float = 0.99,
        damping: float = 1e-4,
        exploration: float = 1e-2,
        device: str = "cpu",
        discrete_actions: bool = True,
    ):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.gamma = gamma  # RL discount
        self.damping = damping  # Belavkin γ
        self.exploration = exploration  # Belavkin β
        self.device = device
        self.discrete_actions = discrete_actions

        # Q-network
        self.q_network = nn.Sequential(
            nn.Linear(state_dim + action_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
        ).to(device)

        # Target network
        self.target_network = nn.Sequential(
            nn.Linear(state_dim + action_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
        ).to(device)
        self.target_network.load_state_dict(self.q_network.state_dict())

        # Belavkin-inspired optimizer
        from belavkin_ml.optimizer import BelavkinOptimizer

        self.optimizer = BelavkinOptimizer(
            self.q_network.parameters(),
            lr=lr,
            gamma=damping,
            beta=exploration,
        )

        # Replay buffer
        self.buffer = deque(maxlen=100000)

    def select_action(self, state: torch.Tensor, epsilon: float = 0.1) -> torch.Tensor:
        """Select action using epsilon-greedy."""
        if random.random() < epsilon:
            if self.discrete_actions:
                return torch.randint(0, self.action_dim, (1,)).squeeze()
            else:
                return torch.randn(self.action_dim, device=self.device)

        with torch.no_grad():
            if self.discrete_actions:
                q_values = []
                for a in range(self.action_dim):
                    action_onehot = F.one_hot(
                        torch.tensor([a]), num_classes=self.action_dim
                    ).float().to(self.device)
                    state_action = torch.cat([state.unsqueeze(0), action_onehot], dim=-1)
                    q = self.q_network(state_action)
                    q_values.append(q.item())

                return torch.tensor(np.argmax(q_values))
            else:
                # For continuous actions, use policy gradient (simplified)
                action = torch.zeros(self.action_dim, device=self.device)
                return action

    def store_transition(
        self,
        state: torch.Tensor,
        action: torch.Tensor,
        reward: float,
        next_state: torch.Tensor,
        done: bool,
    ):
        """Store transition in replay buffer."""
        self.buffer.append((state, action, reward, next_state, done))

    def train(self, batch_size: int = 64) -> Dict[str, float]:
        """Train Q-network using Belavkin optimizer."""
        if len(self.buffer) < batch_size:
            return {}

        # Sample batch
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)

        states = torch.stack([s.to(self.device) for s in states])
        actions = torch.stack([a.to(self.device) for a in actions])
        rewards = torch.tensor(rewards, dtype=torch.float32, device=self.device)
        next_states = torch.stack([s.to(self.device) for s in next_states])
        dones = torch.tensor(dones, dtype=torch.float32, device=self.device)

        # One-hot encode actions if discrete
        if self.discrete_actions:
            actions_onehot = F.one_hot(actions.long(), num_classes=self.action_dim).float()
        else:
            actions_onehot = actions

        # Current Q-values
        state_actions = torch.cat([states, actions_onehot], dim=-1)
        current_q = self.q_network(state_actions).squeeze()

        # Target Q-values
        with torch.no_grad():
            if self.discrete_actions:
                # Max over actions for next state
                next_q_values = []
                for a in range(self.action_dim):
                    next_action_onehot = F.one_hot(
                        torch.full((batch_size,), a, dtype=torch.long),
                        num_classes=self.action_dim,
                    ).float().to(self.device)
                    next_state_actions = torch.cat([next_states, next_action_onehot], dim=-1)
                    next_q = self.target_network(next_state_actions).squeeze()
                    next_q_values.append(next_q)

                next_q_values = torch.stack(next_q_values, dim=0)
                max_next_q = torch.max(next_q_values, dim=0)[0]
            else:
                # Simplified for continuous
                next_actions_onehot = torch.zeros(batch_size, self.action_dim, device=self.device)
                next_state_actions = torch.cat([next_states, next_actions_onehot], dim=-1)
                max_next_q = self.target_network(next_state_actions).squeeze()

            target_q = rewards + self.gamma * max_next_q * (1 - dones)

        # Q-learning loss
        loss = F.mse_loss(current_q, target_q)

        # Update with Belavkin optimizer
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return {"q_loss": loss.item()}

    def update_target_network(self):
        """Update target network."""
        self.target_network.load_state_dict(self.q_network.state_dict())
