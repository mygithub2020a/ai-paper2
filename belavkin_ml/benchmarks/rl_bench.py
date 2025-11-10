"""
RL benchmarking framework for Belavkin agents.
"""

import gymnasium as gym
import torch
import numpy as np
from typing import Dict, List, Any, Optional, Tuple
from pathlib import Path
import json
from tqdm import tqdm
import time

from belavkin_ml.rl.agents import BelavkinAgent, BelavkinDQN, BelavkinPPO
from belavkin_ml.rl.environments import NoisyGridWorld, NoisyPendulum


class RLBenchmark:
    """
    Framework for benchmarking RL agents.

    Args:
        env_factory (callable): Function that creates environment
        max_episodes (int): Maximum training episodes
        max_steps_per_episode (int): Max steps per episode
        eval_every (int): Evaluate every N episodes
        eval_episodes (int): Number of evaluation episodes
        device (str): Device for computation
    """

    def __init__(
        self,
        env_factory,
        max_episodes: int = 1000,
        max_steps_per_episode: int = 200,
        eval_every: int = 50,
        eval_episodes: int = 10,
        device: str = 'cpu',
    ):
        self.env_factory = env_factory
        self.max_episodes = max_episodes
        self.max_steps_per_episode = max_steps_per_episode
        self.eval_every = eval_every
        self.eval_episodes = eval_episodes
        self.device = device

    def train_agent(
        self,
        agent,
        seed: int = 42,
        verbose: bool = True,
    ) -> Dict[str, Any]:
        """
        Train agent on environment.

        Args:
            agent: RL agent
            seed: Random seed
            verbose: Print progress

        Returns:
            results: Training metrics
        """
        env = self.env_factory()

        # Set seeds
        np.random.seed(seed)
        torch.manual_seed(seed)

        results = {
            'episode_rewards': [],
            'episode_lengths': [],
            'eval_rewards': [],
            'eval_episodes': [],
            'training_time': 0.0,
        }

        start_time = time.time()

        iterator = tqdm(range(self.max_episodes), desc="Training") if verbose else range(self.max_episodes)

        for episode in iterator:
            # Reset environment and agent
            obs, _ = env.reset(seed=seed + episode)
            agent.reset()

            episode_reward = 0.0
            episode_length = 0

            for step in range(self.max_steps_per_episode):
                # Select action
                if isinstance(agent, BelavkinDQN):
                    action = agent.select_action(torch.tensor(obs, dtype=torch.float32), training=True)
                    belief_before = agent.filter.belief.get_probabilities()
                elif isinstance(agent, BelavkinPPO):
                    action, log_prob = agent.select_action(torch.tensor(obs, dtype=torch.float32), training=True)
                    belief = agent.filter.belief.get_probabilities()

                # Take action
                next_obs, reward, terminated, truncated, info = env.step(action)
                done = terminated or truncated

                # Update belief
                agent.update_belief(
                    action=action,
                    observation=torch.tensor(next_obs, dtype=torch.float32),
                    reward=reward,
                )

                # Store transition
                if isinstance(agent, BelavkinDQN):
                    belief_after = agent.filter.belief.get_probabilities()
                    agent.store_transition(belief_before, action, reward, belief_after, done)

                    # Train
                    loss = agent.train_step()

                elif isinstance(agent, BelavkinPPO):
                    value = agent.value_network(belief.unsqueeze(0)).item()
                    agent.store_transition(belief, action, reward, log_prob, value)

                episode_reward += reward
                episode_length += 1

                obs = next_obs

                if done:
                    break

            # Train PPO at end of episode
            if isinstance(agent, BelavkinPPO):
                policy_loss, value_loss = agent.train_episode()

            # Record episode metrics
            results['episode_rewards'].append(episode_reward)
            results['episode_lengths'].append(episode_length)

            # Periodic evaluation
            if episode % self.eval_every == 0:
                eval_reward = self.evaluate_agent(agent, self.eval_episodes, seed=seed)
                results['eval_rewards'].append(eval_reward)
                results['eval_episodes'].append(episode)

                if verbose:
                    iterator.set_postfix({
                        'reward': f'{episode_reward:.2f}',
                        'eval': f'{eval_reward:.2f}',
                    })

            # Update target network for DQN
            if isinstance(agent, BelavkinDQN) and episode % 10 == 0:
                agent.update_target_network()

        results['training_time'] = time.time() - start_time

        env.close()

        return results

    def evaluate_agent(
        self,
        agent,
        n_episodes: int,
        seed: int = 42,
    ) -> float:
        """
        Evaluate agent performance.

        Args:
            agent: RL agent
            n_episodes: Number of evaluation episodes
            seed: Random seed

        Returns:
            mean_reward: Average reward over episodes
        """
        env = self.env_factory()

        total_rewards = []

        for ep in range(n_episodes):
            obs, _ = env.reset(seed=seed + ep)
            agent.reset()

            episode_reward = 0.0

            for step in range(self.max_steps_per_episode):
                # Select action (greedy)
                if isinstance(agent, BelavkinDQN):
                    action = agent.select_action(
                        torch.tensor(obs, dtype=torch.float32),
                        training=False
                    )
                elif isinstance(agent, BelavkinPPO):
                    action, _ = agent.select_action(
                        torch.tensor(obs, dtype=torch.float32),
                        training=False
                    )

                # Take action
                next_obs, reward, terminated, truncated, info = env.step(action)
                done = terminated or truncated

                # Update belief
                agent.update_belief(
                    action=action,
                    observation=torch.tensor(next_obs, dtype=torch.float32),
                    reward=reward,
                )

                episode_reward += reward
                obs = next_obs

                if done:
                    break

            total_rewards.append(episode_reward)

        env.close()

        return np.mean(total_rewards)


def run_rl_comparison(
    benchmark: RLBenchmark,
    agents: Dict[str, Any],
    n_seeds: int = 3,
    save_dir: Optional[Path] = None,
    verbose: bool = True,
) -> Dict[str, List[Dict[str, Any]]]:
    """
    Run comparison across multiple RL agents.

    Args:
        benchmark: RLBenchmark instance
        agents: Dict mapping agent name to agent factory
        n_seeds: Number of random seeds
        save_dir: Directory to save results
        verbose: Print progress

    Returns:
        all_results: Results for all agents and seeds
    """
    all_results = {}

    for agent_name, agent_factory in agents.items():
        if verbose:
            print(f"\n{'='*60}")
            print(f"Training {agent_name}")
            print(f"{'='*60}")

        agent_results = []

        for seed in range(n_seeds):
            if verbose:
                print(f"\nSeed {seed + 1}/{n_seeds}")

            # Create fresh agent
            agent = agent_factory()

            # Train
            result = benchmark.train_agent(
                agent=agent,
                seed=seed,
                verbose=verbose,
            )

            result['agent_name'] = agent_name
            result['seed'] = seed

            agent_results.append(result)

        all_results[agent_name] = agent_results

    # Save results
    if save_dir:
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)

        with open(save_dir / 'rl_results.json', 'w') as f:
            serializable_results = {}
            for agent_name, agent_results in all_results.items():
                serializable_results[agent_name] = [
                    {k: (v if not isinstance(v, (np.ndarray, torch.Tensor)) else
                         v.tolist() if isinstance(v, np.ndarray) else v.cpu().tolist())
                     for k, v in result.items()}
                    for result in agent_results
                ]
            json.dump(serializable_results, f, indent=2)

        if verbose:
            print(f"\nResults saved to {save_dir / 'rl_results.json'}")

    return all_results
