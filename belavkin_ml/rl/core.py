"""
Core Belavkin filtering framework for RL.

Implements quantum state estimation and filtering adapted for
reinforcement learning in partially observable environments.
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Tuple, Optional, Dict, Any
from dataclasses import dataclass


@dataclass
class QuantumState:
    """
    Quantum state representation for RL.

    In the classical limit, this reduces to a probability distribution
    over hidden states. The density matrix ρ encodes:
    - Diagonal elements: Probabilities of states
    - Off-diagonal elements: Quantum coherences (if using full quantum model)

    Args:
        density_matrix (torch.Tensor): Density matrix ρ of shape [d, d]
        is_pure (bool): Whether state is pure (can be represented as vector)
    """
    density_matrix: torch.Tensor
    is_pure: bool = False

    def __post_init__(self):
        """Validate quantum state properties."""
        # Check Hermiticity
        if not torch.allclose(self.density_matrix, self.density_matrix.conj().T, atol=1e-6):
            raise ValueError("Density matrix must be Hermitian")

        # Check trace = 1
        trace = torch.trace(self.density_matrix)
        if not torch.isclose(trace, torch.tensor(1.0), atol=1e-6):
            # Normalize
            self.density_matrix = self.density_matrix / trace

        # Check positive semi-definite
        eigvals = torch.linalg.eigvalsh(self.density_matrix)
        if torch.any(eigvals < -1e-6):
            raise ValueError("Density matrix must be positive semi-definite")

    @property
    def dimension(self) -> int:
        """State space dimension."""
        return self.density_matrix.shape[0]

    def get_probabilities(self) -> torch.Tensor:
        """Get probability distribution (diagonal of density matrix)."""
        return torch.diag(self.density_matrix).real

    def sample_state(self) -> int:
        """Sample a state from the distribution."""
        probs = self.get_probabilities()
        return torch.multinomial(probs, num_samples=1).item()

    def entropy(self) -> float:
        """Compute von Neumann entropy: S = -Tr(ρ log ρ)."""
        eigvals = torch.linalg.eigvalsh(self.density_matrix)
        # Filter out zero eigenvalues
        eigvals = eigvals[eigvals > 1e-10]
        return -(eigvals * torch.log(eigvals)).sum().item()

    @staticmethod
    def uniform(dimension: int, device: str = 'cpu') -> 'QuantumState':
        """Create uniform quantum state."""
        rho = torch.eye(dimension, device=device) / dimension
        return QuantumState(rho, is_pure=False)

    @staticmethod
    def pure(state_idx: int, dimension: int, device: str = 'cpu') -> 'QuantumState':
        """Create pure state |i⟩⟨i|."""
        rho = torch.zeros(dimension, dimension, device=device)
        rho[state_idx, state_idx] = 1.0
        return QuantumState(rho, is_pure=True)


class BelavkinFilter:
    """
    Belavkin quantum filter for RL.

    Implements belief state updates using quantum filtering principles.
    In the classical limit, this reduces to Bayesian filtering.

    The general Belavkin equation:
        dρ_t = -i[H_t, ρ_t]dt + L[ρ_t]dt + M[ρ_t]dy_t

    where:
        - ρ_t: Belief state (density matrix)
        - H_t: Hamiltonian encoding rewards and dynamics
        - L[ρ]: Lindblad superoperator (dissipation)
        - M[ρ]dy_t: Measurement update term
        - dy_t: Innovation process (observation - expectation)

    Args:
        state_dim (int): Dimension of state space
        obs_dim (int): Dimension of observation space
        hbar (float): Effective Planck constant (ℏ=0 → classical limit)
        dt (float): Time step for discretization
        device (str): Device for computation
    """

    def __init__(
        self,
        state_dim: int,
        obs_dim: int,
        hbar: float = 0.1,
        dt: float = 0.01,
        device: str = 'cpu',
    ):
        self.state_dim = state_dim
        self.obs_dim = obs_dim
        self.hbar = hbar
        self.dt = dt
        self.device = device

        # Initialize belief to uniform distribution
        self.belief = QuantumState.uniform(state_dim, device=device)

    def predict(
        self,
        belief: QuantumState,
        action: int,
        transition_model: Optional[torch.Tensor] = None,
        reward_model: Optional[torch.Tensor] = None,
    ) -> QuantumState:
        """
        Prediction step: Evolve belief forward given action.

        Implements:
            ρ_{t+1} = T(ρ_t | a) - i[H(a), ρ_t]dt + L[ρ_t]dt

        Args:
            belief: Current belief state
            action: Action taken
            transition_model: Transition tensor T[s', s, a]
            reward_model: Reward tensor R[s, a]

        Returns:
            predicted_belief: Predicted belief state
        """
        rho = belief.density_matrix

        # Apply transition model (if available)
        if transition_model is not None:
            # Transition: ρ' = Σ_s T[·, s, a] ρ[s, s] (classical part)
            probs = torch.diag(rho)
            new_probs = torch.einsum('ijk,j->i', transition_model, probs)
            rho_new = torch.diag(new_probs)
        else:
            rho_new = rho.clone()

        # Hamiltonian evolution (if reward model available)
        if reward_model is not None and self.hbar > 0:
            # H = reward_model[:, action] (reward as Hamiltonian)
            H = torch.diag(reward_model[:, action])

            # Commutator: -i[H, ρ]
            commutator = -1j * (H @ rho_new - rho_new @ H) / self.hbar
            rho_new = rho_new + commutator * self.dt

        # Ensure Hermiticity and normalization
        rho_new = 0.5 * (rho_new + rho_new.conj().T)
        rho_new = rho_new / torch.trace(rho_new)

        return QuantumState(rho_new.real)

    def update(
        self,
        belief: QuantumState,
        observation: torch.Tensor,
        observation_model: Optional[torch.Tensor] = None,
    ) -> QuantumState:
        """
        Update step: Incorporate observation using Belavkin measurement update.

        Implements:
            ρ_{new} = (L_y ρ L_y†) / Tr(L_y ρ L_y†)

        where L_y is the measurement operator for observation y.

        Args:
            belief: Prior belief state
            observation: Observed vector (or index)
            observation_model: Observation model O[o, s]

        Returns:
            updated_belief: Posterior belief state
        """
        rho = belief.density_matrix

        if observation_model is None:
            # No observation, return prior
            return belief

        # Bayesian update (classical)
        if isinstance(observation, int):
            # Discrete observation
            # P(s | o) ∝ P(o | s) P(s)
            likelihood = observation_model[observation, :]  # P(o | s)
            prior = torch.diag(rho)  # P(s)

            posterior = likelihood * prior
            posterior = posterior / posterior.sum()

            rho_new = torch.diag(posterior)

        else:
            # Continuous observation (using projection)
            # For now, use simple Bayesian update
            # This can be extended to full quantum measurement

            # Compute likelihood for each state
            # Assume Gaussian observation: P(o|s) ∝ exp(-||o - μ_s||²)
            # observation_model should be [state_dim, obs_dim] (means)

            diffs = observation.unsqueeze(0) - observation_model  # [state_dim, obs_dim]
            log_likelihood = -0.5 * (diffs ** 2).sum(dim=1)  # [state_dim]
            likelihood = torch.softmax(log_likelihood, dim=0)

            prior = torch.diag(rho)
            posterior = likelihood * prior
            posterior = posterior / posterior.sum()

            rho_new = torch.diag(posterior)

        return QuantumState(rho_new)

    def filter_step(
        self,
        action: int,
        observation: torch.Tensor,
        transition_model: Optional[torch.Tensor] = None,
        observation_model: Optional[torch.Tensor] = None,
        reward_model: Optional[torch.Tensor] = None,
    ) -> Tuple[QuantumState, float]:
        """
        Complete filter step: predict + update.

        Args:
            action: Action taken
            observation: Observation received
            transition_model: Transition dynamics
            observation_model: Observation model
            reward_model: Reward model

        Returns:
            new_belief, innovation
        """
        # Predict
        predicted_belief = self.predict(
            self.belief,
            action,
            transition_model=transition_model,
            reward_model=reward_model,
        )

        # Update
        updated_belief = self.update(
            predicted_belief,
            observation,
            observation_model=observation_model,
        )

        # Compute innovation (KL divergence or similar)
        innovation = self._compute_innovation(predicted_belief, updated_belief)

        self.belief = updated_belief

        return updated_belief, innovation

    def _compute_innovation(
        self,
        prior: QuantumState,
        posterior: QuantumState,
    ) -> float:
        """
        Compute innovation (information gain) from observation.

        Uses relative entropy: D(posterior || prior)

        Args:
            prior: Predicted belief
            posterior: Updated belief

        Returns:
            innovation: KL divergence
        """
        p_prior = prior.get_probabilities() + 1e-10
        p_post = posterior.get_probabilities() + 1e-10

        kl = (p_post * (torch.log(p_post) - torch.log(p_prior))).sum()

        return kl.item()

    def reset(self, initial_belief: Optional[QuantumState] = None):
        """Reset filter to initial state."""
        if initial_belief is None:
            self.belief = QuantumState.uniform(self.state_dim, self.device)
        else:
            self.belief = initial_belief


class LowRankBelavkinFilter(BelavkinFilter):
    """
    Low-rank approximation of Belavkin filter for scalability.

    Maintains belief as mixture of K pure states:
        ρ ≈ Σ_i w_i |ψ_i⟩⟨ψ_i|

    This reduces memory from O(d²) to O(Kd).

    Args:
        state_dim: State space dimension
        obs_dim: Observation dimension
        n_particles: Number of particles/pure states (K)
        hbar: Effective Planck constant
        dt: Time step
        device: Device
    """

    def __init__(
        self,
        state_dim: int,
        obs_dim: int,
        n_particles: int = 100,
        hbar: float = 0.1,
        dt: float = 0.01,
        device: str = 'cpu',
    ):
        super().__init__(state_dim, obs_dim, hbar, dt, device)

        self.n_particles = n_particles

        # Initialize particles uniformly
        self.particles = torch.randint(0, state_dim, (n_particles,), device=device)
        self.weights = torch.ones(n_particles, device=device) / n_particles

    def get_belief_distribution(self) -> torch.Tensor:
        """Get probability distribution from particles."""
        probs = torch.zeros(self.state_dim, device=self.device)

        for particle, weight in zip(self.particles, self.weights):
            probs[particle] += weight

        return probs

    def predict(self, belief, action, transition_model=None, reward_model=None):
        """Particle-based prediction."""
        if transition_model is not None:
            # Sample next state for each particle
            new_particles = []

            for particle in self.particles:
                transition_probs = transition_model[:, particle, action]
                next_state = torch.multinomial(transition_probs, num_samples=1).item()
                new_particles.append(next_state)

            self.particles = torch.tensor(new_particles, device=self.device)

        # Weights unchanged in prediction
        return self.belief

    def update(self, belief, observation, observation_model=None):
        """Particle-based update (reweighting)."""
        if observation_model is None:
            return belief

        # Reweight particles based on observation likelihood
        if isinstance(observation, int):
            likelihood = observation_model[observation, self.particles]
        else:
            # Gaussian likelihood
            diffs = observation.unsqueeze(0) - observation_model[self.particles]
            log_likelihood = -0.5 * (diffs ** 2).sum(dim=1)
            likelihood = torch.exp(log_likelihood)

        self.weights = self.weights * likelihood
        self.weights = self.weights / self.weights.sum()

        # Resample if effective sample size is low
        ess = 1.0 / (self.weights ** 2).sum()

        if ess < self.n_particles / 2:
            self._resample()

        # Update full belief representation
        probs = self.get_belief_distribution()
        self.belief = QuantumState(torch.diag(probs))

        return self.belief

    def _resample(self):
        """Resample particles (systematic resampling)."""
        indices = torch.multinomial(
            self.weights,
            num_samples=self.n_particles,
            replacement=True,
        )

        self.particles = self.particles[indices]
        self.weights = torch.ones(self.n_particles, device=self.device) / self.n_particles
