"""
Utility functions for RL environments and training.
"""

import torch
import numpy as np
from typing import Dict, List, Tuple, Any
from collections import deque
import random


class ReplayBuffer:
    """
    Experience replay buffer for off-policy RL algorithms.
    """

    def __init__(self, capacity: int = 100000):
        """
        Initialize replay buffer.

        Args:
            capacity: Maximum number of transitions to store
        """
        self.buffer = deque(maxlen=capacity)

    def push(
        self,
        state: np.ndarray,
        action: np.ndarray,
        reward: float,
        next_state: np.ndarray,
        done: bool,
    ):
        """Add transition to buffer."""
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size: int) -> Dict[str, torch.Tensor]:
        """
        Sample a batch of transitions.

        Args:
            batch_size: Number of transitions to sample

        Returns:
            Dictionary of batched tensors
        """
        batch = random.sample(self.buffer, batch_size)

        states, actions, rewards, next_states, dones = zip(*batch)

        return {
            'state': torch.FloatTensor(np.array(states)),
            'action': torch.FloatTensor(np.array(actions)),
            'reward': torch.FloatTensor(rewards),
            'next_state': torch.FloatTensor(np.array(next_states)),
            'done': torch.FloatTensor(dones),
        }

    def __len__(self) -> int:
        return len(self.buffer)


def collect_episode(
    env: Any,
    agent: Any,
    max_steps: int = 1000,
    render: bool = False,
    training: bool = True,
) -> Tuple[float, int, List[Dict]]:
    """
    Collect one episode of experience.

    Args:
        env: Environment
        agent: RL agent
        max_steps: Maximum steps per episode
        render: Whether to render
        training: Whether agent is in training mode

    Returns:
        Tuple of (total_reward, episode_length, transitions)
    """
    state = env.reset()
    if isinstance(state, tuple):
        state = state[0]  # Handle gymnasium API

    total_reward = 0.0
    episode_length = 0
    transitions = []

    for step in range(max_steps):
        # Select action
        state_tensor = torch.FloatTensor(state)
        action = agent.select_action(state_tensor, training=training)

        if isinstance(action, torch.Tensor):
            action_np = action.cpu().numpy()
        else:
            action_np = action

        # Environment step
        result = env.step(action_np)
        if len(result) == 4:  # Old gym API
            next_state, reward, done, info = result
            truncated = False
        else:  # New gymnasium API
            next_state, reward, terminated, truncated, info = result
            done = terminated or truncated

        if render:
            env.render()

        # Store transition
        transitions.append({
            'state': state,
            'action': action_np,
            'reward': reward,
            'next_state': next_state,
            'done': done,
        })

        # Update agent belief
        if hasattr(agent, 'update_belief'):
            agent.update_belief(
                observation=state_tensor,
                action=action,
                reward=reward,
                next_observation=torch.FloatTensor(next_state),
                done=done,
            )

        total_reward += reward
        episode_length += 1

        state = next_state

        if done:
            break

    return total_reward, episode_length, transitions


def evaluate_agent(
    env: Any,
    agent: Any,
    num_episodes: int = 10,
    max_steps: int = 1000,
) -> Dict[str, float]:
    """
    Evaluate agent performance.

    Args:
        env: Environment
        agent: RL agent
        num_episodes: Number of evaluation episodes
        max_steps: Maximum steps per episode

    Returns:
        Dictionary of evaluation metrics
    """
    rewards = []
    lengths = []

    for _ in range(num_episodes):
        total_reward, episode_length, _ = collect_episode(
            env=env,
            agent=agent,
            max_steps=max_steps,
            render=False,
            training=False,
        )
        rewards.append(total_reward)
        lengths.append(episode_length)

    return {
        'mean_reward': np.mean(rewards),
        'std_reward': np.std(rewards),
        'mean_length': np.mean(lengths),
        'std_length': np.std(lengths),
        'min_reward': np.min(rewards),
        'max_reward': np.max(rewards),
    }
