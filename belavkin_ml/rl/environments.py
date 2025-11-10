"""
Toy RL environments with partial observability for testing Belavkin agents.

Includes:
- Noisy Gridworld: Partially observable with observation noise
- Noisy Pendulum: Continuous control with measurement uncertainty
"""

import gymnasium as gym
import numpy as np
import torch
from typing import Tuple, Optional, Dict, Any
from gymnasium import spaces


class NoisyGridWorld(gym.Env):
    """
    Gridworld with noisy observations.

    Agent must navigate to a goal in a grid while receiving noisy observations
    about its position.

    Args:
        grid_size (int): Size of square grid
        noise_prob (float): Probability of observation noise
        obs_noise_std (float): Standard deviation of Gaussian observation noise
        sparse_reward (bool): Use sparse (only at goal) or dense rewards
        seed (int): Random seed
    """

    def __init__(
        self,
        grid_size: int = 5,
        noise_prob: float = 0.2,
        obs_noise_std: float = 0.5,
        sparse_reward: bool = True,
        seed: Optional[int] = None,
    ):
        super().__init__()

        self.grid_size = grid_size
        self.noise_prob = noise_prob
        self.obs_noise_std = obs_noise_std
        self.sparse_reward = sparse_reward

        # State space: (x, y) position
        self.state_dim = grid_size * grid_size
        self.observation_space = spaces.Box(
            low=0, high=grid_size-1, shape=(2,), dtype=np.float32
        )
        self.action_space = spaces.Discrete(4)  # up, down, left, right

        # Goal position (bottom-right corner)
        self.goal_pos = np.array([grid_size - 1, grid_size - 1])

        # Current state
        self.agent_pos = None

        if seed is not None:
            self.np_random = np.random.RandomState(seed)
        else:
            self.np_random = np.random.RandomState()

    def reset(self, seed: Optional[int] = None, options: Optional[dict] = None) -> Tuple[np.ndarray, dict]:
        """Reset environment."""
        if seed is not None:
            self.np_random = np.random.RandomState(seed)

        # Random start position (not at goal)
        while True:
            self.agent_pos = self.np_random.randint(0, self.grid_size, size=2)
            if not np.array_equal(self.agent_pos, self.goal_pos):
                break

        observation = self._get_observation()

        return observation, {}

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, dict]:
        """
        Take action in environment.

        Actions:
            0: up (decrease y)
            1: down (increase y)
            2: left (decrease x)
            3: right (increase x)
        """
        # Execute action
        new_pos = self.agent_pos.copy()

        if action == 0:  # up
            new_pos[1] = max(0, new_pos[1] - 1)
        elif action == 1:  # down
            new_pos[1] = min(self.grid_size - 1, new_pos[1] + 1)
        elif action == 2:  # left
            new_pos[0] = max(0, new_pos[0] - 1)
        elif action == 3:  # right
            new_pos[0] = min(self.grid_size - 1, new_pos[0] + 1)

        self.agent_pos = new_pos

        # Compute reward
        done = np.array_equal(self.agent_pos, self.goal_pos)

        if self.sparse_reward:
            reward = 1.0 if done else 0.0
        else:
            # Dense reward: negative Manhattan distance
            dist = np.abs(self.agent_pos - self.goal_pos).sum()
            reward = -dist / (2 * self.grid_size)

        # Get noisy observation
        observation = self._get_observation()

        return observation, reward, done, False, {}

    def _get_observation(self) -> np.ndarray:
        """
        Get noisy observation of current position.

        With probability noise_prob, return random position.
        Otherwise, return true position plus Gaussian noise.
        """
        if self.np_random.random() < self.noise_prob:
            # Completely random observation
            obs = self.np_random.uniform(0, self.grid_size, size=2)
        else:
            # True position plus Gaussian noise
            obs = self.agent_pos.astype(np.float32) + \
                  self.np_random.normal(0, self.obs_noise_std, size=2)

            # Clip to valid range
            obs = np.clip(obs, 0, self.grid_size - 1)

        return obs.astype(np.float32)

    def get_true_state(self) -> np.ndarray:
        """Get true state (for debugging)."""
        return self.agent_pos.copy()

    def state_to_index(self, state: np.ndarray) -> int:
        """Convert (x, y) state to flat index."""
        return state[1] * self.grid_size + state[0]

    def index_to_state(self, index: int) -> np.ndarray:
        """Convert flat index to (x, y) state."""
        y = index // self.grid_size
        x = index % self.grid_size
        return np.array([x, y])


class NoisyPendulum(gym.Env):
    """
    Inverted pendulum with noisy observations.

    Based on classic pendulum problem but with measurement noise.

    State: [θ, θ̇] (angle, angular velocity)
    Observation: Noisy measurements of [cos(θ), sin(θ), θ̇]
    Action: Torque

    Args:
        obs_noise_std (float): Observation noise standard deviation
        dt (float): Time step
        g (float): Gravity constant
        m (float): Pendulum mass
        l (float): Pendulum length
        seed (int): Random seed
    """

    def __init__(
        self,
        obs_noise_std: float = 0.1,
        dt: float = 0.05,
        g: float = 10.0,
        m: float = 1.0,
        l: float = 1.0,
        max_speed: float = 8.0,
        max_torque: float = 2.0,
        seed: Optional[int] = None,
    ):
        super().__init__()

        self.obs_noise_std = obs_noise_std
        self.dt = dt
        self.g = g
        self.m = m
        self.l = l
        self.max_speed = max_speed
        self.max_torque = max_torque

        # State: [θ, θ̇]
        self.state = None

        # Observation space: [cos(θ), sin(θ), θ̇] + noise
        self.observation_space = spaces.Box(
            low=np.array([-1.1, -1.1, -max_speed * 1.1]),
            high=np.array([1.1, 1.1, max_speed * 1.1]),
            dtype=np.float32,
        )

        # Action space: continuous torque
        self.action_space = spaces.Box(
            low=-max_torque,
            high=max_torque,
            shape=(1,),
            dtype=np.float32,
        )

        if seed is not None:
            self.np_random = np.random.RandomState(seed)
        else:
            self.np_random = np.random.RandomState()

    def reset(self, seed: Optional[int] = None, options: Optional[dict] = None) -> Tuple[np.ndarray, dict]:
        """Reset to random initial state."""
        if seed is not None:
            self.np_random = np.random.RandomState(seed)

        # Random initial angle and angular velocity
        theta = self.np_random.uniform(-np.pi, np.pi)
        theta_dot = self.np_random.uniform(-1.0, 1.0)

        self.state = np.array([theta, theta_dot], dtype=np.float32)

        observation = self._get_observation()

        return observation, {}

    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, dict]:
        """
        Step dynamics forward.

        Equation of motion:
            θ̈ = (3g)/(2l) * sin(θ) + (3)/(ml²) * u

        where u is the applied torque.
        """
        theta, theta_dot = self.state
        u = np.clip(action[0], -self.max_torque, self.max_torque)

        # Compute acceleration
        theta_ddot = (3 * self.g) / (2 * self.l) * np.sin(theta) + \
                     (3.0) / (self.m * self.l ** 2) * u

        # Euler integration
        theta_dot = theta_dot + theta_ddot * self.dt
        theta_dot = np.clip(theta_dot, -self.max_speed, self.max_speed)
        theta = theta + theta_dot * self.dt

        # Normalize angle to [-π, π]
        theta = ((theta + np.pi) % (2 * np.pi)) - np.pi

        self.state = np.array([theta, theta_dot], dtype=np.float32)

        # Compute reward (cost to minimize)
        # Reward is higher when pendulum is upright (θ = 0) with low velocity
        cost = theta ** 2 + 0.1 * theta_dot ** 2 + 0.001 * u ** 2
        reward = -cost

        # Episode terminates after fixed time (handled externally)
        done = False

        observation = self._get_observation()

        return observation, reward, done, False, {'true_state': self.state.copy()}

    def _get_observation(self) -> np.ndarray:
        """
        Get noisy observation.

        Returns: [cos(θ) + noise, sin(θ) + noise, θ̇ + noise]
        """
        theta, theta_dot = self.state

        # True observation
        obs = np.array([
            np.cos(theta),
            np.sin(theta),
            theta_dot,
        ], dtype=np.float32)

        # Add Gaussian noise
        noise = self.np_random.normal(0, self.obs_noise_std, size=3)
        noisy_obs = obs + noise

        return noisy_obs.astype(np.float32)

    def get_true_state(self) -> np.ndarray:
        """Get true state (for debugging)."""
        return self.state.copy()


class NoisyCartPole(gym.Env):
    """
    CartPole with observation noise.

    Classic cart-pole balancing task with noisy observations.

    State: [x, ẋ, θ, θ̇] (position, velocity, angle, angular velocity)
    Observation: Noisy measurements of state
    Action: Discrete (left/right force)

    Args:
        obs_noise_std (float): Observation noise standard deviation
        seed (int): Random seed
    """

    def __init__(
        self,
        obs_noise_std: float = 0.1,
        seed: Optional[int] = None,
    ):
        super().__init__()

        self.obs_noise_std = obs_noise_std

        # Use gym's CartPole as base
        self._base_env = gym.make('CartPole-v1')

        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(4,),
            dtype=np.float32,
        )
        self.action_space = spaces.Discrete(2)

        if seed is not None:
            self.np_random = np.random.RandomState(seed)
        else:
            self.np_random = np.random.RandomState()

    def reset(self, seed: Optional[int] = None, options: Optional[dict] = None) -> Tuple[np.ndarray, dict]:
        """Reset environment."""
        if seed is not None:
            self.np_random = np.random.RandomState(seed)

        true_obs, info = self._base_env.reset(seed=seed)
        noisy_obs = self._add_noise(true_obs)

        return noisy_obs, info

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, dict]:
        """Step environment."""
        true_obs, reward, terminated, truncated, info = self._base_env.step(action)
        noisy_obs = self._add_noise(true_obs)

        info['true_state'] = true_obs

        return noisy_obs, reward, terminated, truncated, info

    def _add_noise(self, observation: np.ndarray) -> np.ndarray:
        """Add Gaussian noise to observation."""
        noise = self.np_random.normal(0, self.obs_noise_std, size=observation.shape)
        return (observation + noise).astype(np.float32)

    def get_true_state(self) -> np.ndarray:
        """Get true state."""
        return self._base_env.unwrapped.state

    def render(self, mode: str = 'human'):
        """Render environment."""
        return self._base_env.render()

    def close(self):
        """Close environment."""
        self._base_env.close()
