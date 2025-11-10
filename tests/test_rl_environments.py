"""
Unit tests for RL environments.
"""

import pytest
import torch
import numpy as np
from belavkin_ml.rl.environments import (
    NoisyGridworld,
    NoisyPendulum,
    NoisyCartPole,
    TabularMDP,
)


def test_noisy_gridworld_creation():
    """Test gridworld creation."""
    env = NoisyGridworld(size=5, observation_noise=0.5)

    assert env.state_dim == 2
    assert env.action_dim == 4
    assert env.observation_dim == 2


def test_noisy_gridworld_reset():
    """Test gridworld reset."""
    env = NoisyGridworld(size=5)

    state, observation = env.reset()

    assert state.shape[0] == 2
    assert observation.shape[0] == 2


def test_noisy_gridworld_step():
    """Test gridworld step."""
    env = NoisyGridworld(size=5)

    state, observation = env.reset()

    # Take a step
    next_state, next_obs, reward, done, info = env.step(0)  # up

    assert next_state.shape[0] == 2
    assert next_obs.shape[0] == 2
    assert isinstance(reward, float)
    assert isinstance(done, (bool, np.bool_))


def test_noisy_gridworld_goal():
    """Test reaching goal."""
    env = NoisyGridworld(size=3)

    # Set agent at goal manually
    env.goal_pos = np.array([1, 1])
    env.agent_pos = np.array([1, 1])

    _, _, reward, done, _ = env.step(0)

    assert done  # Should be done
    assert reward > 0  # Should receive goal reward


def test_noisy_pendulum():
    """Test noisy pendulum environment."""
    env = NoisyPendulum(observation_noise=0.1)

    assert env.state_dim == 3
    assert env.action_dim == 1

    state, observation = env.reset()
    assert state.shape[0] == 3
    assert observation.shape[0] == 3

    action = torch.tensor([0.5])
    next_state, next_obs, reward, done, info = env.step(action)

    assert next_state.shape[0] == 3
    assert isinstance(reward, (float, np.floating))

    env.close()


def test_noisy_cartpole():
    """Test noisy CartPole environment."""
    env = NoisyCartPole(observation_noise=0.1)

    assert env.state_dim == 4
    assert env.action_dim == 2

    state, observation = env.reset()
    assert state.shape[0] == 4
    assert observation.shape[0] == 4

    next_state, next_obs, reward, done, info = env.step(0)

    assert next_state.shape[0] == 4
    assert isinstance(reward, (float, np.floating))

    env.close()


def test_cartpole_observation_mask():
    """Test CartPole with observation mask."""
    # Hide velocity components
    mask = np.array([True, False, True, False])

    env = NoisyCartPole(observation_noise=0.0, observation_mask=mask)

    state, observation = env.reset()

    # Masked components should be zero
    assert observation[1] == 0.0
    assert observation[3] == 0.0


def test_tabular_mdp():
    """Test tabular MDP."""
    env = TabularMDP(n_states=10, n_actions=4, observation_noise=0.1)

    assert env.state_dim == 10
    assert env.action_dim == 4

    state, observation = env.reset()
    assert state.shape[0] == 10
    assert torch.sum(state) == 1.0  # One-hot

    next_state, next_obs, reward, done, info = env.step(0)

    assert next_state.shape[0] == 10
    assert isinstance(reward, (float, np.floating))


def test_environment_reproducibility():
    """Test that environments are reproducible."""
    np.random.seed(42)
    env1 = NoisyGridworld(size=5)
    state1, obs1 = env1.reset()

    np.random.seed(42)
    env2 = NoisyGridworld(size=5)
    state2, obs2 = env2.reset()

    # Should produce same initial states
    assert torch.allclose(state1, state2)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
