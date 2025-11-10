"""
Noisy Gridworld Environment

A simple gridworld with observation noise - perfect for testing Belavkin filtering.

Features:
- Grid of size N x N
- Agent starts at random position
- Goal at fixed position
- Observations are noisy (Gaussian noise added to position)
- Partial observability through limited observation radius

This tests the Belavkin agent's ability to maintain beliefs under uncertainty.
"""

import numpy as np
from typing import Tuple, Optional, Dict


class NoisyGridworld:
    """
    Noisy gridworld environment with partial observability.
    """

    def __init__(
        self,
        grid_size: int = 10,
        observation_noise: float = 0.5,
        observation_radius: int = 2,
        goal_reward: float = 10.0,
        step_penalty: float = -0.1,
        random_seed: Optional[int] = None,
    ):
        """
        Initialize noisy gridworld.

        Args:
            grid_size: Size of grid (N x N)
            observation_noise: Std dev of Gaussian noise on observations
            observation_radius: How far agent can "see"
            goal_reward: Reward for reaching goal
            step_penalty: Penalty per step
            random_seed: Random seed
        """
        self.grid_size = grid_size
        self.observation_noise = observation_noise
        self.observation_radius = observation_radius
        self.goal_reward = goal_reward
        self.step_penalty = step_penalty

        if random_seed is not None:
            np.random.seed(random_seed)

        # State space: (x, y) position
        self.state_dim = 2

        # Action space: [dx, dy] continuous in [-1, 1]
        self.action_dim = 2

        # Observation: noisy position + local features
        # Observation = [noisy_x, noisy_y, distance_to_goal, angle_to_goal]
        self.observation_dim = 4

        # Fixed goal position (top-right corner)
        self.goal_pos = np.array([grid_size - 1.0, grid_size - 1.0])

        # Current state
        self.agent_pos = None

        # Episode statistics
        self.steps = 0
        self.max_steps = 200

    def reset(self) -> np.ndarray:
        """
        Reset environment.

        Returns:
            Initial observation
        """
        # Random start position (not at goal)
        self.agent_pos = np.random.uniform(0, self.grid_size, size=2)

        # Ensure not starting at goal
        while np.linalg.norm(self.agent_pos - self.goal_pos) < 1.0:
            self.agent_pos = np.random.uniform(0, self.grid_size, size=2)

        self.steps = 0

        return self._get_observation()

    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, Dict]:
        """
        Take a step in the environment.

        Args:
            action: Action [dx, dy] in [-1, 1]

        Returns:
            Tuple of (observation, reward, done, info)
        """
        # Clip action
        action = np.clip(action, -1.0, 1.0)

        # Move agent (scale action for reasonable step size)
        self.agent_pos += action * 0.5

        # Clip to grid boundaries
        self.agent_pos = np.clip(self.agent_pos, 0, self.grid_size)

        # Compute reward
        distance_to_goal = np.linalg.norm(self.agent_pos - self.goal_pos)

        if distance_to_goal < 0.5:
            # Reached goal
            reward = self.goal_reward
            done = True
        else:
            # Step penalty + negative reward based on distance
            reward = self.step_penalty - 0.01 * distance_to_goal
            done = False

        self.steps += 1

        # Check max steps
        if self.steps >= self.max_steps:
            done = True

        observation = self._get_observation()

        info = {
            'true_position': self.agent_pos.copy(),
            'distance_to_goal': distance_to_goal,
            'steps': self.steps,
        }

        return observation, reward, done, info

    def _get_observation(self) -> np.ndarray:
        """
        Get noisy observation of current state.

        Returns:
            Observation vector [noisy_x, noisy_y, dist_to_goal, angle_to_goal]
        """
        # Add Gaussian noise to position
        noisy_pos = self.agent_pos + np.random.normal(
            0, self.observation_noise, size=2
        )

        # Compute distance and angle to goal
        delta = self.goal_pos - self.agent_pos
        distance = np.linalg.norm(delta)
        angle = np.arctan2(delta[1], delta[0])

        # Construct observation
        observation = np.array([
            noisy_pos[0] / self.grid_size,  # Normalized noisy x
            noisy_pos[1] / self.grid_size,  # Normalized noisy y
            distance / (self.grid_size * np.sqrt(2)),  # Normalized distance
            angle / np.pi,  # Normalized angle
        ])

        return observation

    def render(self):
        """Simple text rendering."""
        grid = np.zeros((self.grid_size, self.grid_size), dtype=str)
        grid[:, :] = '.'

        # Mark goal
        gx, gy = int(self.goal_pos[0]), int(self.goal_pos[1])
        if 0 <= gx < self.grid_size and 0 <= gy < self.grid_size:
            grid[gy, gx] = 'G'

        # Mark agent
        ax, ay = int(self.agent_pos[0]), int(self.agent_pos[1])
        if 0 <= ax < self.grid_size and 0 <= ay < self.grid_size:
            grid[ay, ax] = 'A'

        print('\n' + '-' * (self.grid_size * 2 + 1))
        for row in grid:
            print('|' + ' '.join(row) + '|')
        print('-' * (self.grid_size * 2 + 1))


def test_noisy_gridworld():
    """Test noisy gridworld environment."""
    print("Testing Noisy Gridworld Environment...")

    env = NoisyGridworld(grid_size=10, observation_noise=0.3)

    # Run a few random episodes
    for episode in range(3):
        obs = env.reset()
        print(f"\nEpisode {episode + 1}")
        print(f"  Initial observation: {obs}")

        total_reward = 0
        for step in range(50):
            # Random action
            action = np.random.uniform(-1, 1, size=2)

            obs, reward, done, info = env.step(action)
            total_reward += reward

            if done:
                print(f"  Episode finished at step {step + 1}")
                print(f"  Total reward: {total_reward:.2f}")
                print(f"  Final distance to goal: {info['distance_to_goal']:.2f}")
                break

    print("\n✓ Noisy gridworld test passed!\n")


if __name__ == "__main__":
    test_noisy_gridworld()
