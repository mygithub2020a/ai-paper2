"""
Belief State Representation for Belavkin RL

Manages belief states in partially observable environments using
quantum filtering principles.

The belief state evolves according to:
    dρ/dt = -i[H, ρ] + L[ρ] + measurement_update(observation)

For computational tractability, we use:
1. Low-rank approximation: ρ ≈ Σᵢ wᵢ |ψᵢ⟩⟨ψᵢ|
2. Neural density matrix: ρ parameterized by neural network
3. Particle filter approximation
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Tuple, Optional, List
from dataclasses import dataclass


@dataclass
class BeliefStateConfig:
    """Configuration for belief state representation."""
    state_dim: int
    action_dim: int
    observation_dim: int
    representation: str = "low_rank"  # "low_rank", "neural", "particle"
    rank: int = 10  # For low-rank approximation
    n_particles: int = 100  # For particle filter
    hidden_dim: int = 128  # For neural representation
    measurement_strength: float = 0.1
    diffusion_coeff: float = 0.01


class BelavkinBeliefState:
    """
    Belief state management using Belavkin filtering.

    Maintains a representation of the agent's uncertainty about the true
    environment state, updated using quantum filtering principles.

    Args:
        config: Configuration for belief state
        device: torch device
    """

    def __init__(self, config: BeliefStateConfig, device: str = "cpu"):
        self.config = config
        self.device = device

        if config.representation == "low_rank":
            self._init_low_rank()
        elif config.representation == "neural":
            self._init_neural()
        elif config.representation == "particle":
            self._init_particle()
        else:
            raise ValueError(f"Unknown representation: {config.representation}")

    def _init_low_rank(self):
        """Initialize low-rank belief state."""
        # Represent ρ ≈ Σᵢ wᵢ |ψᵢ⟩⟨ψᵢ| with k components
        self.weights = torch.ones(self.config.rank, device=self.device) / self.config.rank
        self.states = torch.randn(
            self.config.rank, self.config.state_dim, device=self.device
        )
        # Normalize states
        self.states = self.states / torch.norm(self.states, dim=1, keepdim=True)

    def _init_neural(self):
        """Initialize neural density matrix representation."""
        # Neural network that outputs density matrix elements
        self.density_net = nn.Sequential(
            nn.Linear(self.config.observation_dim, self.config.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.config.hidden_dim, self.config.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.config.hidden_dim, self.config.state_dim ** 2),
        ).to(self.device)

        # Running observation buffer
        self.obs_buffer = torch.zeros(
            self.config.observation_dim, device=self.device
        )

    def _init_particle(self):
        """Initialize particle filter representation."""
        self.particles = torch.randn(
            self.config.n_particles,
            self.config.state_dim,
            device=self.device
        )
        self.particle_weights = torch.ones(
            self.config.n_particles, device=self.device
        ) / self.config.n_particles

    def predict(self, action: torch.Tensor, dt: float = 1.0) -> None:
        """
        Prediction step: evolve belief state under action.

        Implements: dρ/dt = -i[H(action), ρ] + diffusion_term

        Args:
            action: Action taken
            dt: Time step
        """
        if self.config.representation == "low_rank":
            self._predict_low_rank(action, dt)
        elif self.config.representation == "neural":
            self._predict_neural(action, dt)
        elif self.config.representation == "particle":
            self._predict_particle(action, dt)

    def _predict_low_rank(self, action: torch.Tensor, dt: float):
        """Prediction for low-rank representation."""
        # Simple dynamics: states evolve with action-dependent drift
        action_effect = action.unsqueeze(0).expand(self.config.rank, -1)

        # Add action-dependent drift (simplified Hamiltonian evolution)
        self.states = self.states + dt * action_effect[:, :self.config.state_dim]

        # Add diffusion (quantum noise)
        noise = torch.randn_like(self.states) * np.sqrt(dt * self.config.diffusion_coeff)
        self.states = self.states + noise

        # Renormalize
        self.states = self.states / torch.norm(self.states, dim=1, keepdim=True)

    def _predict_neural(self, action: torch.Tensor, dt: float):
        """Prediction for neural representation."""
        # Neural prediction is implicit in the network
        # Add action effect to observation buffer
        if action.shape[0] < self.config.observation_dim:
            action_padded = torch.zeros(self.config.observation_dim, device=self.device)
            action_padded[:action.shape[0]] = action
            self.obs_buffer = self.obs_buffer * 0.9 + action_padded * 0.1

    def _predict_particle(self, action: torch.Tensor, dt: float):
        """Prediction for particle filter."""
        # Propagate particles with action
        action_effect = action.unsqueeze(0).expand(self.config.n_particles, -1)
        self.particles = self.particles + dt * action_effect[:, :self.config.state_dim]

        # Add process noise
        noise = torch.randn_like(self.particles) * np.sqrt(dt * self.config.diffusion_coeff)
        self.particles = self.particles + noise

    def update(self, observation: torch.Tensor, reward: float) -> None:
        """
        Update step: incorporate observation using Belavkin measurement update.

        Implements: measurement_update term from Belavkin equation

        Args:
            observation: Observed state/feature
            reward: Reward signal (can influence update)
        """
        if self.config.representation == "low_rank":
            self._update_low_rank(observation, reward)
        elif self.config.representation == "neural":
            self._update_neural(observation, reward)
        elif self.config.representation == "particle":
            self._update_particle(observation, reward)

    def _update_low_rank(self, observation: torch.Tensor, reward: float):
        """Update for low-rank representation."""
        # Compute likelihood of observation for each component
        # Using simple Gaussian likelihood
        obs_expanded = observation.unsqueeze(0).expand(self.config.rank, -1)

        # Likelihood based on distance to observation
        distances = torch.norm(
            self.states[:, :observation.shape[0]] - obs_expanded,
            dim=1
        )
        likelihoods = torch.exp(-distances ** 2 / (2 * self.config.measurement_strength))

        # Bayes update: w_i ∝ w_i * p(obs | ψ_i)
        self.weights = self.weights * likelihoods
        self.weights = self.weights / (torch.sum(self.weights) + 1e-8)

        # Measurement backaction: shift states toward observation
        shift = observation.unsqueeze(0).expand(self.config.rank, -1)
        self.states[:, :observation.shape[0]] += self.config.measurement_strength * (
            shift - self.states[:, :observation.shape[0]]
        )

        # Incorporate reward signal (quantum control influence)
        self.weights = self.weights * (1 + reward * 0.1)
        self.weights = self.weights / (torch.sum(self.weights) + 1e-8)

    def _update_neural(self, observation: torch.Tensor, reward: float):
        """Update for neural representation."""
        # Update observation buffer
        if observation.shape[0] <= self.config.observation_dim:
            self.obs_buffer[:observation.shape[0]] = observation
        else:
            self.obs_buffer = observation[:self.config.observation_dim]

    def _update_particle(self, observation: torch.Tensor, reward: float):
        """Update for particle filter."""
        # Compute weights based on observation likelihood
        obs_expanded = observation.unsqueeze(0).expand(self.config.n_particles, -1)

        distances = torch.norm(
            self.particles[:, :observation.shape[0]] - obs_expanded,
            dim=1
        )
        likelihoods = torch.exp(-distances ** 2 / (2 * self.config.measurement_strength))

        # Update particle weights
        self.particle_weights = self.particle_weights * likelihoods
        self.particle_weights = self.particle_weights / (torch.sum(self.particle_weights) + 1e-8)

        # Resample if effective sample size is low
        ess = 1.0 / torch.sum(self.particle_weights ** 2)
        if ess < self.config.n_particles / 2:
            self._resample_particles()

    def _resample_particles(self):
        """Resample particles based on weights."""
        indices = torch.multinomial(
            self.particle_weights,
            self.config.n_particles,
            replacement=True
        )
        self.particles = self.particles[indices]
        self.particle_weights = torch.ones(
            self.config.n_particles, device=self.device
        ) / self.config.n_particles

    def get_mean_state(self) -> torch.Tensor:
        """Get mean of belief distribution."""
        if self.config.representation == "low_rank":
            return torch.sum(
                self.weights.unsqueeze(1) * self.states, dim=0
            )
        elif self.config.representation == "neural":
            # Get density matrix from network
            rho_flat = self.density_net(self.obs_buffer)
            rho = rho_flat.reshape(self.config.state_dim, self.config.state_dim)
            # Return diagonal (mean state)
            return torch.diagonal(rho)
        elif self.config.representation == "particle":
            return torch.sum(
                self.particle_weights.unsqueeze(1) * self.particles, dim=0
            )

    def get_uncertainty(self) -> float:
        """Get measure of belief uncertainty (entropy-like)."""
        if self.config.representation == "low_rank":
            # Entropy of weight distribution
            entropy = -torch.sum(self.weights * torch.log(self.weights + 1e-8))
            return entropy.item()
        elif self.config.representation == "neural":
            # Use variance of diagonal elements
            rho_flat = self.density_net(self.obs_buffer)
            rho = rho_flat.reshape(self.config.state_dim, self.config.state_dim)
            return torch.var(torch.diagonal(rho)).item()
        elif self.config.representation == "particle":
            # Entropy of particle weights
            entropy = -torch.sum(
                self.particle_weights * torch.log(self.particle_weights + 1e-8)
            )
            return entropy.item()

    def reset(self):
        """Reset belief state to uniform prior."""
        if self.config.representation == "low_rank":
            self._init_low_rank()
        elif self.config.representation == "neural":
            self._init_neural()
        elif self.config.representation == "particle":
            self._init_particle()
