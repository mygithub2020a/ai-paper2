"""
Simple environments for testing Belavkin RL agents.

Environments:
1. NoisyGridworld: Partially observable gridworld with noisy observations
2. NoisyPendulum: Pendulum with observation noise
3. NoisyCartPole: CartPole with partial observability
"""

import torch
import numpy as np
from typing import Tuple, Optional
import gymnasium as gym


class NoisyGridworld:
    """
    Simple gridworld with noisy observations.

    The agent must navigate to a goal while only receiving noisy observations
    of its position. This tests the belief state management.

    Args:
        size: Grid size (size x size)
        observation_noise: Standard deviation of observation noise
        goal_reward: Reward for reaching goal
        step_penalty: Penalty for each step
    """

    def __init__(
        self,
        size: int = 5,
        observation_noise: float = 0.5,
        goal_reward: float = 10.0,
        step_penalty: float = -0.1,
    ):
        self.size = size
        self.observation_noise = observation_noise
        self.goal_reward = goal_reward
        self.step_penalty = step_penalty

        # State space: (x, y) position
        self.state_dim = 2
        # Action space: up, down, left, right
        self.action_dim = 4
        # Observation space: noisy (x, y)
        self.observation_dim = 2

        # Internal state
        self.agent_pos = None
        self.goal_pos = None

        self.reset()

    def reset(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Reset environment.

        Returns:
            state: True agent position
            observation: Noisy observation
        """
        # Random agent position
        self.agent_pos = np.random.randint(0, self.size, size=2)

        # Random goal position (not same as agent)
        self.goal_pos = np.random.randint(0, self.size, size=2)
        while np.array_equal(self.agent_pos, self.goal_pos):
            self.goal_pos = np.random.randint(0, self.size, size=2)

        state = torch.tensor(self.agent_pos, dtype=torch.float32)
        observation = self._get_observation()

        return state, observation

    def _get_observation(self) -> torch.Tensor:
        """Get noisy observation of agent position."""
        noise = np.random.randn(2) * self.observation_noise
        noisy_pos = self.agent_pos + noise
        # Clip to valid range
        noisy_pos = np.clip(noisy_pos, 0, self.size - 1)
        return torch.tensor(noisy_pos, dtype=torch.float32)

    def step(
        self, action: int
    ) -> Tuple[torch.Tensor, torch.Tensor, float, bool, dict]:
        """
        Take a step in the environment.

        Args:
            action: 0=up, 1=down, 2=left, 3=right

        Returns:
            state: New true state
            observation: Noisy observation
            reward: Reward signal
            done: Whether episode is done
            info: Additional info
        """
        # Move agent
        if action == 0:  # up
            self.agent_pos[1] = max(0, self.agent_pos[1] - 1)
        elif action == 1:  # down
            self.agent_pos[1] = min(self.size - 1, self.agent_pos[1] + 1)
        elif action == 2:  # left
            self.agent_pos[0] = max(0, self.agent_pos[0] - 1)
        elif action == 3:  # right
            self.agent_pos[0] = min(self.size - 1, self.agent_pos[0] + 1)

        # Check if goal reached
        done = np.array_equal(self.agent_pos, self.goal_pos)

        # Compute reward
        if done:
            reward = self.goal_reward
        else:
            reward = self.step_penalty

        state = torch.tensor(self.agent_pos, dtype=torch.float32)
        observation = self._get_observation()

        info = {
            "true_position": self.agent_pos.copy(),
            "goal_position": self.goal_pos.copy(),
        }

        return state, observation, reward, done, info

    def render(self):
        """Simple text rendering."""
        grid = np.full((self.size, self.size), ".")
        grid[self.goal_pos[1], self.goal_pos[0]] = "G"
        grid[self.agent_pos[1], self.agent_pos[0]] = "A"

        print("\nGridworld:")
        for row in grid:
            print(" ".join(row))
        print()


class NoisyPendulum:
    """
    Pendulum with observation noise.

    Wraps the standard pendulum environment but adds Gaussian noise to
    observations to create partial observability.

    Args:
        observation_noise: Standard deviation of observation noise
    """

    def __init__(self, observation_noise: float = 0.1):
        self.env = gym.make("Pendulum-v1")
        self.observation_noise = observation_noise

        self.state_dim = 3  # cos(theta), sin(theta), theta_dot
        self.action_dim = 1
        self.observation_dim = 3

    def reset(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """Reset environment."""
        state, _ = self.env.reset()
        state = torch.tensor(state, dtype=torch.float32)

        # Add observation noise
        noise = torch.randn_like(state) * self.observation_noise
        observation = state + noise

        return state, observation

    def step(
        self, action: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, float, bool, dict]:
        """Take a step."""
        # Convert to numpy
        action_np = action.cpu().numpy()
        if action_np.ndim == 0:
            action_np = np.array([action_np])

        # Step environment
        state, reward, terminated, truncated, info = self.env.step(action_np)
        done = terminated or truncated

        state = torch.tensor(state, dtype=torch.float32)

        # Add observation noise
        noise = torch.randn_like(state) * self.observation_noise
        observation = state + noise

        return state, observation, reward, done, info

    def close(self):
        """Close environment."""
        self.env.close()


class NoisyCartPole:
    """
    CartPole with partial observability through noisy observations.

    Args:
        observation_noise: Standard deviation of observation noise
        observation_mask: Which state components are visible (default: all)
    """

    def __init__(
        self,
        observation_noise: float = 0.1,
        observation_mask: Optional[np.ndarray] = None,
    ):
        self.env = gym.make("CartPole-v1")
        self.observation_noise = observation_noise

        self.state_dim = 4  # cart position, cart velocity, pole angle, pole velocity
        self.action_dim = 2  # left, right
        self.observation_dim = 4

        # Observation mask: which components are visible
        if observation_mask is None:
            self.observation_mask = np.ones(4, dtype=bool)
        else:
            self.observation_mask = observation_mask

    def reset(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """Reset environment."""
        state, _ = self.env.reset()
        state = torch.tensor(state, dtype=torch.float32)

        observation = self._get_observation(state)

        return state, observation

    def _get_observation(self, state: torch.Tensor) -> torch.Tensor:
        """Get noisy, partially observable observation."""
        # Add noise
        noise = torch.randn_like(state) * self.observation_noise
        noisy_state = state + noise

        # Apply observation mask
        observation = noisy_state.clone()
        observation[~self.observation_mask] = 0.0  # Hide unobservable components

        return observation

    def step(
        self, action: int
    ) -> Tuple[torch.Tensor, torch.Tensor, float, bool, dict]:
        """Take a step."""
        state, reward, terminated, truncated, info = self.env.step(action)
        done = terminated or truncated

        state = torch.tensor(state, dtype=torch.float32)
        observation = self._get_observation(state)

        return state, observation, reward, done, info

    def close(self):
        """Close environment."""
        self.env.close()


class TabularMDP:
    """
    Simple tabular MDP for testing.

    Useful for verifying that algorithms work correctly before scaling up.

    Args:
        n_states: Number of states
        n_actions: Number of actions
        observation_noise: Noise in state observations
    """

    def __init__(
        self, n_states: int = 10, n_actions: int = 4, observation_noise: float = 0.1
    ):
        self.n_states = n_states
        self.n_actions = n_actions
        self.observation_noise = observation_noise

        self.state_dim = n_states  # One-hot representation
        self.action_dim = n_actions
        self.observation_dim = n_states

        # Random transition dynamics: P(s' | s, a)
        self.transitions = np.random.dirichlet(
            np.ones(n_states), size=(n_states, n_actions)
        )

        # Random reward function: R(s, a)
        self.rewards = np.random.randn(n_states, n_actions)

        # Goal state
        self.goal_state = n_states - 1

        self.current_state = None
        self.reset()

    def reset(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """Reset to random initial state."""
        self.current_state = np.random.randint(0, self.n_states)

        state = self._state_to_tensor(self.current_state)
        observation = self._get_observation(state)

        return state, observation

    def _state_to_tensor(self, state_idx: int) -> torch.Tensor:
        """Convert state index to one-hot tensor."""
        state = np.zeros(self.n_states)
        state[state_idx] = 1.0
        return torch.tensor(state, dtype=torch.float32)

    def _get_observation(self, state: torch.Tensor) -> torch.Tensor:
        """Add noise to state observation."""
        noise = torch.randn_like(state) * self.observation_noise
        observation = state + noise
        return observation

    def step(
        self, action: int
    ) -> Tuple[torch.Tensor, torch.Tensor, float, bool, dict]:
        """Take a step in the MDP."""
        # Sample next state according to transition dynamics
        next_state_probs = self.transitions[self.current_state, action]
        next_state = np.random.choice(self.n_states, p=next_state_probs)

        # Get reward
        reward = self.rewards[self.current_state, action]

        # Check if done (reached goal)
        done = next_state == self.goal_state

        # Update state
        self.current_state = next_state

        state = self._state_to_tensor(next_state)
        observation = self._get_observation(state)

        info = {"true_state": next_state}

        return state, observation, reward, done, info
