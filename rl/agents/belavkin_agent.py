"""
Belavkin RL Agent - Deep RL using quantum filtering principles.

This agent uses the Belavkin quantum filtering equation as the core
update mechanism for policy improvement and value estimation.

Key innovations:
1. Quantum-inspired exploration via measurement uncertainty
2. Adaptive learning via quantum filtering dynamics
3. Stochastic policy updates based on quantum state evolution
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Tuple, List, Optional, Dict
from collections import deque
import copy

from ..models.networks import PolicyValueNetwork
from belavkin_optimizer import BelavkinOptimizer


class BelavkinAgent:
    """
    Belavkin-based RL agent.

    Uses quantum filtering principles for policy and value updates.

    Args:
        network: Policy-value network
        lr: Base learning rate
        gamma: Discount factor for returns
        belavkin_gamma: Adaptive damping factor (from Belavkin equation)
        belavkin_beta: Stochastic exploration factor
        device: torch device
    """

    def __init__(
        self,
        network: nn.Module,
        lr: float = 1e-3,
        gamma: float = 0.99,
        belavkin_gamma: float = 1e-4,
        belavkin_beta: float = 1e-5,
        entropy_coef: float = 0.01,
        value_coef: float = 0.5,
        max_grad_norm: float = 10.0,
        device: Optional[torch.device] = None,
    ):
        self.network = network
        self.gamma = gamma
        self.entropy_coef = entropy_coef
        self.value_coef = value_coef
        self.max_grad_norm = max_grad_norm
        self.device = device if device is not None else torch.device('cpu')

        self.network.to(self.device)

        # Use Belavkin optimizer
        self.optimizer = BelavkinOptimizer(
            network.parameters(),
            lr=lr,
            gamma=belavkin_gamma,
            beta=belavkin_beta,
            adaptive_gamma=True,
        )

        # Experience buffer
        self.states = []
        self.actions = []
        self.rewards = []
        self.values = []
        self.log_probs = []
        self.dones = []

    def select_action(
        self,
        state: np.ndarray,
        deterministic: bool = False,
    ) -> Tuple[int, float, float]:
        """
        Select action using current policy.

        Args:
            state: Current state
            deterministic: If True, select argmax action

        Returns:
            action: Selected action
            log_prob: Log probability of action
            value: Estimated state value
        """
        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            policy_logits, value = self.network(state_tensor)

            if deterministic:
                action = policy_logits.argmax(dim=1).item()
                log_prob = F.log_softmax(policy_logits, dim=1)[0, action].item()
            else:
                # Sample from policy
                probs = F.softmax(policy_logits, dim=1)
                action_dist = torch.distributions.Categorical(probs)
                action = action_dist.sample().item()
                log_prob = action_dist.log_prob(torch.tensor(action)).item()

            value = value.item()

        return action, log_prob, value

    def store_transition(
        self,
        state: np.ndarray,
        action: int,
        reward: float,
        value: float,
        log_prob: float,
        done: bool,
    ):
        """Store transition in buffer."""
        self.states.append(state)
        self.actions.append(action)
        self.rewards.append(reward)
        self.values.append(value)
        self.log_probs.append(log_prob)
        self.dones.append(done)

    def compute_returns(
        self,
        next_value: float = 0.0,
        use_gae: bool = True,
        gae_lambda: float = 0.95,
    ) -> Tuple[List[float], List[float]]:
        """
        Compute returns and advantages.

        Args:
            next_value: Value of next state (for bootstrapping)
            use_gae: Whether to use Generalized Advantage Estimation
            gae_lambda: GAE lambda parameter

        Returns:
            returns: List of returns
            advantages: List of advantages
        """
        returns = []
        advantages = []

        if use_gae:
            gae = 0
            next_value = next_value
            for t in reversed(range(len(self.rewards))):
                if t == len(self.rewards) - 1:
                    next_non_terminal = 1.0 - self.dones[t]
                    next_val = next_value
                else:
                    next_non_terminal = 1.0 - self.dones[t]
                    next_val = self.values[t + 1]

                delta = (
                    self.rewards[t]
                    + self.gamma * next_val * next_non_terminal
                    - self.values[t]
                )
                gae = delta + self.gamma * gae_lambda * next_non_terminal * gae
                advantages.insert(0, gae)
                returns.insert(0, gae + self.values[t])
        else:
            R = next_value
            for t in reversed(range(len(self.rewards))):
                R = self.rewards[t] + self.gamma * R * (1.0 - self.dones[t])
                returns.insert(0, R)
                advantages.insert(0, R - self.values[t])

        return returns, advantages

    def update(
        self,
        next_value: float = 0.0,
        use_gae: bool = True,
    ) -> Dict[str, float]:
        """
        Update policy and value function using stored transitions.

        Returns:
            Dictionary of training metrics
        """
        if len(self.states) == 0:
            return {}

        # Compute returns and advantages
        returns, advantages = self.compute_returns(next_value, use_gae)

        # Convert to tensors
        states = torch.FloatTensor(np.array(self.states)).to(self.device)
        actions = torch.LongTensor(self.actions).to(self.device)
        returns = torch.FloatTensor(returns).to(self.device)
        advantages = torch.FloatTensor(advantages).to(self.device)
        old_log_probs = torch.FloatTensor(self.log_probs).to(self.device)

        # Normalize advantages
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        # Forward pass
        policy_logits, values = self.network(states)
        values = values.squeeze()

        # Policy loss (with quantum-inspired exploration)
        log_probs = F.log_softmax(policy_logits, dim=1)
        action_log_probs = log_probs.gather(1, actions.unsqueeze(1)).squeeze()

        # Standard policy gradient
        policy_loss = -(action_log_probs * advantages.detach()).mean()

        # Entropy regularization (quantum uncertainty)
        entropy = -(log_probs.exp() * log_probs).sum(dim=1).mean()
        policy_loss = policy_loss - self.entropy_coef * entropy

        # Value loss
        value_loss = F.mse_loss(values, returns)

        # Total loss
        loss = policy_loss + self.value_coef * value_loss

        # Backward pass with Belavkin optimizer
        self.optimizer.zero_grad()
        loss.backward()

        # Gradient clipping
        if self.max_grad_norm > 0:
            nn.utils.clip_grad_norm_(self.network.parameters(), self.max_grad_norm)

        self.optimizer.step()

        # Clear buffer
        self.states = []
        self.actions = []
        self.rewards = []
        self.values = []
        self.log_probs = []
        self.dones = []

        # Return metrics
        return {
            'loss': loss.item(),
            'policy_loss': policy_loss.item(),
            'value_loss': value_loss.item(),
            'entropy': entropy.item(),
        }


class BelavkinAlphaZeroAgent:
    """
    AlphaZero-style agent with Belavkin optimizer.

    Combines MCTS with Belavkin-based policy improvement.
    """

    def __init__(
        self,
        network: nn.Module,
        lr: float = 1e-3,
        belavkin_gamma: float = 1e-4,
        belavkin_beta: float = 1e-5,
        mcts_simulations: int = 800,
        c_puct: float = 1.0,
        temperature: float = 1.0,
        device: Optional[torch.device] = None,
    ):
        self.network = network
        self.mcts_simulations = mcts_simulations
        self.c_puct = c_puct
        self.temperature = temperature
        self.device = device if device is not None else torch.device('cpu')

        self.network.to(self.device)

        # Use Belavkin optimizer
        self.optimizer = BelavkinOptimizer(
            network.parameters(),
            lr=lr,
            gamma=belavkin_gamma,
            beta=belavkin_beta,
            adaptive_gamma=True,
        )

        # Replay buffer
        self.replay_buffer = deque(maxlen=10000)

    def select_action_mcts(
        self,
        state: np.ndarray,
        legal_actions: List[int],
        temperature: Optional[float] = None,
    ) -> Tuple[int, np.ndarray]:
        """
        Select action using MCTS.

        Args:
            state: Current state
            legal_actions: List of legal actions
            temperature: Temperature for action selection

        Returns:
            action: Selected action
            visit_counts: Visit counts for all actions (MCTS policy)
        """
        if temperature is None:
            temperature = self.temperature

        # Run MCTS simulations
        # (Simplified version - full MCTS implementation would be more complex)
        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            policy_logits, value = self.network(state_tensor)
            policy = F.softmax(policy_logits, dim=1).cpu().numpy()[0]

        # Mask illegal actions
        legal_mask = np.zeros(len(policy))
        legal_mask[legal_actions] = 1.0
        policy = policy * legal_mask
        policy = policy / (policy.sum() + 1e-8)

        # Temperature-based sampling
        if temperature == 0:
            action = legal_actions[policy[legal_actions].argmax()]
            visit_counts = np.zeros(len(policy))
            visit_counts[action] = 1.0
        else:
            # Apply temperature
            policy = policy ** (1.0 / temperature)
            policy = policy / policy.sum()

            # Sample action
            action = np.random.choice(len(policy), p=policy)
            visit_counts = policy

        return action, visit_counts

    def store_experience(
        self,
        state: np.ndarray,
        mcts_policy: np.ndarray,
        value: float,
    ):
        """Store experience in replay buffer."""
        self.replay_buffer.append((state, mcts_policy, value))

    def train_step(self, batch_size: int = 32) -> Dict[str, float]:
        """
        Perform one training step.

        Args:
            batch_size: Batch size for training

        Returns:
            Dictionary of training metrics
        """
        if len(self.replay_buffer) < batch_size:
            return {}

        # Sample batch
        indices = np.random.choice(len(self.replay_buffer), batch_size, replace=False)
        batch = [self.replay_buffer[i] for i in indices]

        states = torch.FloatTensor(np.array([x[0] for x in batch])).to(self.device)
        target_policies = torch.FloatTensor(np.array([x[1] for x in batch])).to(
            self.device
        )
        target_values = torch.FloatTensor(np.array([x[2] for x in batch])).to(
            self.device
        )

        # Forward pass
        policy_logits, values = self.network(states)
        values = values.squeeze()

        # Policy loss (cross-entropy)
        policy_loss = F.cross_entropy(policy_logits, target_policies.argmax(dim=1))

        # Value loss
        value_loss = F.mse_loss(values, target_values)

        # Total loss
        loss = policy_loss + value_loss

        # Backward pass with Belavkin optimizer
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return {
            'loss': loss.item(),
            'policy_loss': policy_loss.item(),
            'value_loss': value_loss.item(),
        }

    def save(self, path: str):
        """Save agent to file."""
        torch.save({
            'network_state_dict': self.network.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
        }, path)

    def load(self, path: str):
        """Load agent from file."""
        checkpoint = torch.load(path)
        self.network.load_state_dict(checkpoint['network_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
