"""Tests for RL core components."""

import pytest
import torch
import numpy as np
from belavkin_ml.rl.core import QuantumState, BelavkinFilter, LowRankBelavkinFilter


def test_quantum_state_creation():
    """Test quantum state creation."""
    # Uniform state
    state = QuantumState.uniform(dimension=5)

    assert state.dimension == 5
    assert torch.allclose(torch.trace(state.density_matrix), torch.tensor(1.0))

    probs = state.get_probabilities()
    assert torch.allclose(probs, torch.ones(5) / 5)


def test_quantum_state_pure():
    """Test pure state creation."""
    state = QuantumState.pure(state_idx=2, dimension=5)

    assert state.is_pure
    assert state.dimension == 5

    probs = state.get_probabilities()
    expected = torch.zeros(5)
    expected[2] = 1.0
    assert torch.allclose(probs, expected)


def test_quantum_state_entropy():
    """Test entropy computation."""
    # Uniform state has maximum entropy
    uniform_state = QuantumState.uniform(dimension=5)
    entropy = uniform_state.entropy()

    # Should be log(5)
    expected_entropy = np.log(5)
    assert np.isclose(entropy, expected_entropy, rtol=1e-5)

    # Pure state has zero entropy
    pure_state = QuantumState.pure(state_idx=0, dimension=5)
    entropy = pure_state.entropy()
    assert np.isclose(entropy, 0.0, atol=1e-6)


def test_belavkin_filter_initialization():
    """Test Belavkin filter initialization."""
    filter = BelavkinFilter(
        state_dim=10,
        obs_dim=3,
        hbar=0.1,
        dt=0.01,
    )

    assert filter.state_dim == 10
    assert filter.obs_dim == 3
    assert filter.belief is not None


def test_belavkin_filter_predict():
    """Test prediction step."""
    filter = BelavkinFilter(state_dim=5, obs_dim=2)

    # Create simple transition model
    # transition[s', s, a] = probability of s' given s and a
    transition_model = torch.zeros(5, 5, 2)  # 2 actions

    # Deterministic transitions for action 0: s' = (s + 1) mod 5
    for s in range(5):
        transition_model[(s + 1) % 5, s, 0] = 1.0

    # Predict
    belief = filter.predict(
        filter.belief,
        action=0,
        transition_model=transition_model,
    )

    assert belief.dimension == 5
    assert torch.allclose(torch.trace(belief.density_matrix), torch.tensor(1.0))


def test_belavkin_filter_update():
    """Test update step."""
    filter = BelavkinFilter(state_dim=5, obs_dim=1)

    # Create observation model: P(o | s)
    obs_model = torch.eye(5)  # Perfect observations

    # Update with observation 2
    belief = filter.update(
        filter.belief,
        observation=2,
        observation_model=obs_model,
    )

    # Should concentrate probability on state 2
    probs = belief.get_probabilities()
    assert probs[2] > 0.5  # Most probability on observed state


def test_low_rank_filter():
    """Test low-rank approximation."""
    filter = LowRankBelavkinFilter(
        state_dim=10,
        obs_dim=2,
        n_particles=20,
    )

    assert filter.n_particles == 20
    assert len(filter.particles) == 20

    # Get distribution
    probs = filter.get_belief_distribution()
    assert probs.shape == (10,)
    assert torch.allclose(probs.sum(), torch.tensor(1.0))


def test_filter_step():
    """Test complete filter step."""
    filter = BelavkinFilter(state_dim=5, obs_dim=1)

    # Transition model
    transition_model = torch.zeros(5, 5, 2)
    for s in range(5):
        transition_model[(s + 1) % 5, s, 0] = 1.0
        transition_model[s, s, 1] = 1.0

    # Observation model
    obs_model = torch.eye(5)

    # Complete step
    belief, innovation = filter.filter_step(
        action=0,
        observation=2,
        transition_model=transition_model,
        observation_model=obs_model,
    )

    assert belief is not None
    assert isinstance(innovation, float)


def test_filter_reset():
    """Test filter reset."""
    filter = BelavkinFilter(state_dim=5, obs_dim=1)

    # Modify belief
    filter.belief = QuantumState.pure(state_idx=2, dimension=5)

    # Reset
    filter.reset()

    # Should be uniform again
    probs = filter.belief.get_probabilities()
    assert torch.allclose(probs, torch.ones(5) / 5)


if __name__ == '__main__':
    pytest.main([__file__])
