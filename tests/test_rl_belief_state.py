"""
Unit tests for Belavkin RL belief state.
"""

import pytest
import torch
import numpy as np
from belavkin_ml.rl.belief_state import BelavkinBeliefState, BeliefStateConfig


def test_belief_state_initialization():
    """Test belief state initialization."""
    config = BeliefStateConfig(
        state_dim=4,
        action_dim=2,
        observation_dim=4,
        representation='low_rank',
        rank=5,
    )

    belief = BelavkinBeliefState(config)
    assert belief is not None


def test_belief_state_low_rank():
    """Test low-rank belief state."""
    config = BeliefStateConfig(
        state_dim=4,
        action_dim=2,
        observation_dim=4,
        representation='low_rank',
        rank=5,
    )

    belief = BelavkinBeliefState(config)

    # Check initialization
    assert belief.weights.shape[0] == 5
    assert belief.states.shape == (5, 4)

    # Check weights are normalized
    assert torch.abs(torch.sum(belief.weights) - 1.0) < 1e-6


def test_belief_state_predict():
    """Test prediction step."""
    config = BeliefStateConfig(
        state_dim=4,
        action_dim=2,
        observation_dim=4,
        representation='low_rank',
        rank=5,
    )

    belief = BelavkinBeliefState(config)

    action = torch.tensor([1.0, 0.0])

    # Perform prediction
    belief.predict(action, dt=0.1)

    # States should still be valid
    assert belief.weights.shape[0] == 5
    assert belief.states.shape == (5, 4)


def test_belief_state_update():
    """Test update step with observation."""
    config = BeliefStateConfig(
        state_dim=4,
        action_dim=2,
        observation_dim=4,
        representation='low_rank',
        rank=5,
    )

    belief = BelavkinBeliefState(config)

    observation = torch.tensor([1.0, 0.5, -0.5, 0.2])
    reward = 1.0

    # Perform update
    belief.update(observation, reward)

    # Weights should still be normalized
    assert torch.abs(torch.sum(belief.weights) - 1.0) < 1e-5


def test_belief_state_mean():
    """Test getting mean state."""
    config = BeliefStateConfig(
        state_dim=4,
        action_dim=2,
        observation_dim=4,
        representation='low_rank',
        rank=5,
    )

    belief = BelavkinBeliefState(config)

    mean_state = belief.get_mean_state()

    assert mean_state.shape[0] == 4


def test_belief_state_uncertainty():
    """Test uncertainty computation."""
    config = BeliefStateConfig(
        state_dim=4,
        action_dim=2,
        observation_dim=4,
        representation='low_rank',
        rank=5,
    )

    belief = BelavkinBeliefState(config)

    uncertainty = belief.get_uncertainty()

    assert isinstance(uncertainty, float)
    assert uncertainty >= 0


def test_belief_state_reset():
    """Test belief state reset."""
    config = BeliefStateConfig(
        state_dim=4,
        action_dim=2,
        observation_dim=4,
        representation='low_rank',
        rank=5,
    )

    belief = BelavkinBeliefState(config)

    # Modify state
    observation = torch.tensor([1.0, 0.5, -0.5, 0.2])
    belief.update(observation, 1.0)

    initial_mean = belief.get_mean_state()

    # Reset
    belief.reset()

    # State should be different after reset
    reset_mean = belief.get_mean_state()
    assert not torch.allclose(initial_mean, reset_mean, atol=1e-4)


def test_belief_state_particle_filter():
    """Test particle filter representation."""
    config = BeliefStateConfig(
        state_dim=4,
        action_dim=2,
        observation_dim=4,
        representation='particle',
        n_particles=50,
    )

    belief = BelavkinBeliefState(config)

    assert belief.particles.shape == (50, 4)
    assert belief.particle_weights.shape[0] == 50

    # Test prediction and update
    action = torch.tensor([1.0, 0.0])
    belief.predict(action, dt=0.1)

    observation = torch.tensor([1.0, 0.5, -0.5, 0.2])
    belief.update(observation, 1.0)

    mean_state = belief.get_mean_state()
    assert mean_state.shape[0] == 4


def test_belief_state_neural():
    """Test neural density matrix representation."""
    config = BeliefStateConfig(
        state_dim=4,
        action_dim=2,
        observation_dim=4,
        representation='neural',
        hidden_dim=32,
    )

    belief = BelavkinBeliefState(config)

    assert belief.density_net is not None

    # Test operations
    action = torch.tensor([1.0, 0.0])
    belief.predict(action, dt=0.1)

    observation = torch.tensor([1.0, 0.5, -0.5, 0.2])
    belief.update(observation, 1.0)

    mean_state = belief.get_mean_state()
    assert mean_state.shape[0] == 4


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
