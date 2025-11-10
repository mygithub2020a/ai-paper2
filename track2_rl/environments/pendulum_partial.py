"""
Partially Observable Pendulum Environment

Classic pendulum control with partial observability and measurement noise.

State: [theta, theta_dot] (angle and angular velocity)
Observation: Noisy measurements of position (not velocity)
Action: Torque

This tests belief state maintenance for continuous control.
"""

import numpy as np
from typing import Tuple, Optional, Dict


class PartialObservabilityPendulum:
    """
    Pendulum environment with partial observability.
    """

    def __init__(
        self,
        max_speed: float = 8.0,
        max_torque: float = 2.0,
        dt: float = 0.05,
        g: float = 10.0,
        m: float = 1.0,
        l: float = 1.0,
        observation_noise: float = 0.1,
        random_seed: Optional[int] = None,
    ):
        """
        Initialize pendulum.

        Args:
            max_speed: Maximum angular velocity
            max_torque: Maximum applied torque
            dt: Time step
            g: Gravity
            m: Mass
            l: Length
            observation_noise: Std dev of observation noise
            random_seed: Random seed
        """
        self.max_speed = max_speed
        self.max_torque = max_torque
        self.dt = dt
        self.g = g
        self.m = m
        self.l = l
        self.observation_noise = observation_noise

        if random_seed is not None:
            np.random.seed(random_seed)

        # State: [theta, theta_dot]
        self.state_dim = 2

        # Action: [torque]
        self.action_dim = 1

        # Observation: [cos(theta), sin(theta)] + noise
        # Note: velocity is NOT observed (partial observability)
        self.observation_dim = 2

        # Current state
        self.state = None

        # Episode tracking
        self.steps = 0
        self.max_steps = 200

    def reset(self) -> np.ndarray:
        """
        Reset pendulum to random state.

        Returns:
            Initial observation
        """
        # Random initial angle and velocity
        theta = np.random.uniform(-np.pi, np.pi)
        theta_dot = np.random.uniform(-1.0, 1.0)

        self.state = np.array([theta, theta_dot])
        self.steps = 0

        return self._get_observation()

    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, Dict]:
        """
        Simulate one step of pendulum dynamics.

        Args:
            action: Torque to apply

        Returns:
            Tuple of (observation, reward, done, info)
        """
        theta, theta_dot = self.state

        # Clip action
        u = np.clip(action, -self.max_torque, self.max_torque)[0]

        # Pendulum dynamics: theta_ddot = (3g/2l)sin(theta) + (3/ml^2)u
        theta_ddot = (3 * self.g / (2 * self.l)) * np.sin(theta) + \
                     (3.0 / (self.m * self.l ** 2)) * u

        # Euler integration
        theta_dot_new = theta_dot + theta_ddot * self.dt
        theta_dot_new = np.clip(theta_dot_new, -self.max_speed, self.max_speed)

        theta_new = theta + theta_dot_new * self.dt

        # Normalize angle to [-pi, pi]
        theta_new = ((theta_new + np.pi) % (2 * np.pi)) - np.pi

        # Update state
        self.state = np.array([theta_new, theta_dot_new])

        # Compute reward (goal: upright position with low velocity)
        # Reward is higher when close to upright (theta=0) and low velocity
        cost = theta_new ** 2 + 0.1 * theta_dot_new ** 2 + 0.001 * u ** 2
        reward = -cost

        self.steps += 1
        done = self.steps >= self.max_steps

        observation = self._get_observation()

        info = {
            'true_state': self.state.copy(),
            'angle': theta_new,
            'angular_velocity': theta_dot_new,
            'steps': self.steps,
        }

        return observation, reward, done, info

    def _get_observation(self) -> np.ndarray:
        """
        Get partial, noisy observation.

        Returns:
            Observation: [cos(theta) + noise, sin(theta) + noise]
            Note: Angular velocity is NOT observed
        """
        theta, _ = self.state

        # Encode angle as [cos(theta), sin(theta)] with noise
        cos_theta = np.cos(theta) + np.random.normal(0, self.observation_noise)
        sin_theta = np.sin(theta) + np.random.normal(0, self.observation_noise)

        observation = np.array([cos_theta, sin_theta])

        return observation

    def render(self):
        """Simple text rendering."""
        theta, theta_dot = self.state
        print(f"  Angle: {theta:.3f} rad ({np.degrees(theta):.1f}°)")
        print(f"  Velocity: {theta_dot:.3f} rad/s")
        print(f"  Steps: {self.steps}/{self.max_steps}")


def test_partial_pendulum():
    """Test partially observable pendulum."""
    print("Testing Partially Observable Pendulum...")

    env = PartialObservabilityPendulum(observation_noise=0.05)

    # Run a few random episodes
    for episode in range(2):
        obs = env.reset()
        print(f"\nEpisode {episode + 1}")
        print(f"  Initial observation: {obs}")

        total_reward = 0
        for step in range(50):
            # Random action
            action = np.random.uniform(-2, 2, size=1)

            obs, reward, done, info = env.step(action)
            total_reward += reward

            if step % 10 == 0:
                print(f"  Step {step}: reward={reward:.3f}")

            if done:
                print(f"  Episode finished")
                print(f"  Total reward: {total_reward:.2f}")
                break

    print("\n✓ Partial pendulum test passed!\n")


if __name__ == "__main__":
    test_partial_pendulum()
