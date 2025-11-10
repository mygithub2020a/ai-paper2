"""
Belavkin Quantum Filtering Framework for Reinforcement Learning

This module implements a novel RL approach based on Belavkin's quantum
filtering equations. The framework models RL as a continuous quantum
state estimation problem.

Theoretical Framework:
    Standard Belavkin equation:
        dψ_t = [-iH_t - (1/2)L_t*L_t]ψ_t dt + [L_t - ⟨L_t⟩_ψ]ψ_t dy_t

    Adapted to RL:
        - ψ_t (density matrix ρ_t): Agent's belief about environment state
        - H_t = H(s, a): Reward structure and dynamics (action-dependent)
        - L_t: Observation/measurement operators
        - Control u_t: Actions selected by policy

Key Innovation:
    Model the RL problem as optimal quantum state estimation where:
    1. Belief state is represented as density matrix
    2. Actions modify the system Hamiltonian
    3. Observations update belief via quantum filtering
    4. Policy optimizes expected reward under belief uncertainty

References:
    - Belavkin, V.P. (1992). Quantum stochastic calculus
    - Belavkin & Guta (2008). Quantum Stochastics and Information
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Dict, Tuple, Optional, List
from collections import deque
import gymnasium as gym


class DensityMatrix:
    """
    Efficient representation of quantum density matrix for belief states.

    For tractability, we use low-rank approximation:
        ρ ≈ Σᵢ wᵢ |ψᵢ⟩⟨ψᵢ|

    Args:
        state_dim (int): Dimension of state space
        rank (int): Rank of approximation (number of basis states)
    """

    def __init__(self, state_dim: int, rank: int = 10):
        self.state_dim = state_dim
        self.rank = rank

        # Initialize with ensemble of pure states
        # Each column is a state vector |ψᵢ⟩
        self.states = torch.randn(state_dim, rank, dtype=torch.complex64)
        self.states /= torch.norm(self.states, dim=0, keepdim=True)

        # Weights for ensemble
        self.weights = torch.ones(rank) / rank

    def as_matrix(self) -> torch.Tensor:
        """
        Compute full density matrix ρ = Σᵢ wᵢ |ψᵢ⟩⟨ψᵢ|.

        Returns:
            Complex tensor of shape (state_dim, state_dim)
        """
        rho = torch.zeros(self.state_dim, self.state_dim, dtype=torch.complex64)
        for i in range(self.rank):
            psi = self.states[:, i]
            rho += self.weights[i] * torch.outer(psi, psi.conj())
        return rho

    def evolve(self, hamiltonian: torch.Tensor, dt: float = 0.01):
        """
        Unitary evolution under Hamiltonian: ψ → exp(-iHt)ψ.

        Args:
            hamiltonian: Hermitian operator H
            dt: Time step
        """
        # U = exp(-iHdt)
        U = torch.matrix_exp(-1j * hamiltonian * dt)
        self.states = U @ self.states

    def measurement_update(self, measurement_op: torch.Tensor, outcome: float):
        """
        Update density matrix based on measurement outcome.

        Implements collapse: ρ → L ρ L† / Tr(L ρ L†)

        Args:
            measurement_op: Measurement operator L
            outcome: Observed measurement result
        """
        # Apply measurement operator
        for i in range(self.rank):
            psi = self.states[:, i]
            psi_new = measurement_op @ psi
            # Normalize
            norm = torch.norm(psi_new)
            if norm > 1e-8:
                self.states[:, i] = psi_new / norm
                self.weights[i] *= norm**2

        # Renormalize weights
        self.weights /= self.weights.sum()

    def expectation(self, observable: torch.Tensor) -> float:
        """
        Compute expectation value ⟨O⟩ = Tr(ρ O).

        Args:
            observable: Hermitian operator O

        Returns:
            Real expectation value
        """
        rho = self.as_matrix()
        return torch.real(torch.trace(rho @ observable)).item()

    def entropy(self) -> float:
        """
        Compute von Neumann entropy S = -Tr(ρ log ρ).

        Returns:
            Entropy value (non-negative)
        """
        # For pure states: S = 0
        # For mixed states: S > 0
        # Approximate using ensemble weights
        return -torch.sum(self.weights * torch.log(self.weights + 1e-10)).item()


class BelavkinRLAgent:
    """
    Reinforcement learning agent using Belavkin quantum filtering.

    This implements a model-based approach where:
    1. Agent maintains belief state as density matrix
    2. Actions modify system Hamiltonian
    3. Observations update belief via quantum filtering
    4. Policy trained to maximize expected reward

    Args:
        state_dim (int): State space dimension
        action_dim (int): Action space dimension
        rank (int): Rank for density matrix approximation
        gamma (float): Discount factor
        learning_rate (float): Learning rate for policy
    """

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        rank: int = 10,
        gamma: float = 0.99,
        learning_rate: float = 1e-3,
    ):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.rank = rank
        self.gamma = gamma

        # Belief state (density matrix)
        self.belief = DensityMatrix(state_dim, rank)

        # Policy network: maps belief features to action probabilities
        self.policy_net = nn.Sequential(
            nn.Linear(state_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, action_dim),
            nn.Softmax(dim=-1),
        )

        # Value network: estimates V(ρ)
        self.value_net = nn.Sequential(
            nn.Linear(state_dim, 128), nn.ReLU(), nn.Linear(128, 1)
        )

        self.optimizer = torch.optim.Adam(
            list(self.policy_net.parameters()) + list(self.value_net.parameters()),
            lr=learning_rate,
        )

        # Learned components
        self.hamiltonian_net = nn.Linear(
            action_dim, state_dim * state_dim
        )  # H(a)
        self.measurement_net = nn.Linear(
            state_dim, state_dim * state_dim
        )  # L(obs)

        # Experience buffer
        self.memory = deque(maxlen=10000)

    def get_belief_features(self) -> torch.Tensor:
        """
        Extract features from belief state for policy/value networks.

        Returns:
            Feature vector of dimension state_dim
        """
        # Use diagonal of density matrix as features
        rho = self.belief.as_matrix()
        features = torch.real(torch.diag(rho))
        return features

    def select_action(self, training: bool = True) -> int:
        """
        Select action based on current belief state.

        Args:
            training: If True, sample from policy; else take argmax

        Returns:
            Selected action index
        """
        features = self.get_belief_features()
        action_probs = self.policy_net(features)

        if training:
            # Sample action
            action = torch.multinomial(action_probs, 1).item()
        else:
            # Greedy action
            action = torch.argmax(action_probs).item()

        return action

    def get_hamiltonian(self, action: int) -> torch.Tensor:
        """
        Compute action-dependent Hamiltonian H(a).

        Args:
            action: Selected action

        Returns:
            Hermitian matrix of shape (state_dim, state_dim)
        """
        # One-hot encode action
        action_vec = torch.zeros(self.action_dim)
        action_vec[action] = 1.0

        # Generate Hamiltonian from network
        H_flat = self.hamiltonian_net(action_vec)
        H = H_flat.view(self.state_dim, self.state_dim)

        # Ensure Hermitian
        H = (H + H.T) / 2
        return H.to(torch.complex64)

    def get_measurement_operator(self, observation: torch.Tensor) -> torch.Tensor:
        """
        Compute observation-dependent measurement operator L(obs).

        Args:
            observation: Observed state

        Returns:
            Complex matrix of shape (state_dim, state_dim)
        """
        L_flat = self.measurement_net(observation)
        L = L_flat.view(self.state_dim, self.state_dim)
        return L.to(torch.complex64)

    def update_belief(
        self, action: int, observation: torch.Tensor, reward: float, dt: float = 0.1
    ):
        """
        Update belief state using Belavkin filtering.

        Steps:
        1. Hamiltonian evolution based on action
        2. Measurement update based on observation

        Args:
            action: Executed action
            observation: Received observation
            reward: Received reward
            dt: Time step
        """
        # 1. Hamiltonian evolution: ψ → exp(-iH(a)t)ψ
        H = self.get_hamiltonian(action)
        self.belief.evolve(H, dt)

        # 2. Measurement update: ρ → L ρ L† / Tr(L ρ L†)
        L = self.get_measurement_operator(observation)
        self.belief.measurement_update(L, outcome=reward)

    def train_step(self, batch_size: int = 32) -> Dict[str, float]:
        """
        Perform one training step using sampled experience.

        Args:
            batch_size: Number of transitions to sample

        Returns:
            Dictionary of training metrics
        """
        if len(self.memory) < batch_size:
            return {}

        # Sample batch
        batch = np.random.choice(len(self.memory), batch_size, replace=False)
        transitions = [self.memory[i] for i in batch]

        # Unpack transitions
        states = torch.stack([t[0] for t in transitions])
        actions = torch.tensor([t[1] for t in transitions])
        rewards = torch.tensor([t[2] for t in transitions])
        next_states = torch.stack([t[3] for t in transitions])
        dones = torch.tensor([t[4] for t in transitions])

        # Compute policy loss (REINFORCE-style)
        action_probs = self.policy_net(states)
        log_probs = torch.log(
            action_probs.gather(1, actions.unsqueeze(1)) + 1e-10
        ).squeeze()

        # Compute values
        values = self.value_net(states).squeeze()
        next_values = self.value_net(next_states).squeeze()

        # Compute advantages
        targets = rewards + self.gamma * next_values * (1 - dones.float())
        advantages = targets - values

        # Policy gradient loss
        policy_loss = -(log_probs * advantages.detach()).mean()

        # Value loss
        value_loss = nn.MSELoss()(values, targets.detach())

        # Combined loss
        loss = policy_loss + 0.5 * value_loss

        # Optimize
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return {
            'loss': loss.item(),
            'policy_loss': policy_loss.item(),
            'value_loss': value_loss.item(),
            'mean_advantage': advantages.mean().item(),
        }

    def reset(self):
        """Reset belief state to initial distribution."""
        self.belief = DensityMatrix(self.state_dim, self.rank)


class BelavkinRLTrainer:
    """
    Training loop for Belavkin RL agent.

    Args:
        env: Gymnasium environment
        agent: BelavkinRLAgent instance
        n_episodes: Number of training episodes
        max_steps: Maximum steps per episode
    """

    def __init__(
        self,
        env: gym.Env,
        agent: BelavkinRLAgent,
        n_episodes: int = 1000,
        max_steps: int = 200,
    ):
        self.env = env
        self.agent = agent
        self.n_episodes = n_episodes
        self.max_steps = max_steps

        self.episode_rewards = []
        self.episode_lengths = []

    def train(self, log_interval: int = 10):
        """
        Run training loop.

        Args:
            log_interval: Frequency of logging

        Returns:
            Training history
        """
        for episode in range(self.n_episodes):
            # Reset environment and belief
            obs, info = self.env.reset()
            obs = torch.tensor(obs, dtype=torch.float32)
            self.agent.reset()

            episode_reward = 0
            step = 0

            for step in range(self.max_steps):
                # Select action
                action = self.agent.select_action(training=True)

                # Execute in environment
                next_obs, reward, terminated, truncated, info = self.env.step(action)
                done = terminated or truncated
                next_obs = torch.tensor(next_obs, dtype=torch.float32)

                # Update belief using Belavkin filtering
                self.agent.update_belief(action, next_obs, reward)

                # Store transition
                features = self.agent.get_belief_features()
                self.agent.memory.append((features, action, reward, features, done))

                # Train agent
                if len(self.agent.memory) > 32:
                    self.agent.train_step(batch_size=32)

                episode_reward += reward
                obs = next_obs

                if done:
                    break

            self.episode_rewards.append(episode_reward)
            self.episode_lengths.append(step + 1)

            # Logging
            if (episode + 1) % log_interval == 0:
                avg_reward = np.mean(self.episode_rewards[-log_interval:])
                avg_length = np.mean(self.episode_lengths[-log_interval:])
                print(
                    f"Episode {episode+1}/{self.n_episodes} | "
                    f"Avg Reward: {avg_reward:.2f} | "
                    f"Avg Length: {avg_length:.1f}"
                )

        return {
            'episode_rewards': self.episode_rewards,
            'episode_lengths': self.episode_lengths,
        }

    def evaluate(self, n_episodes: int = 10) -> Dict[str, float]:
        """
        Evaluate trained agent.

        Args:
            n_episodes: Number of evaluation episodes

        Returns:
            Evaluation metrics
        """
        eval_rewards = []

        for _ in range(n_episodes):
            obs, info = self.env.reset()
            obs = torch.tensor(obs, dtype=torch.float32)
            self.agent.reset()

            episode_reward = 0

            for _ in range(self.max_steps):
                action = self.agent.select_action(training=False)
                next_obs, reward, terminated, truncated, info = self.env.step(action)
                done = terminated or truncated

                episode_reward += reward

                if done:
                    break

                next_obs = torch.tensor(next_obs, dtype=torch.float32)
                self.agent.update_belief(action, next_obs, reward)
                obs = next_obs

            eval_rewards.append(episode_reward)

        return {
            'mean_reward': np.mean(eval_rewards),
            'std_reward': np.std(eval_rewards),
            'min_reward': np.min(eval_rewards),
            'max_reward': np.max(eval_rewards),
        }
