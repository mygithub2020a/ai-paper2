"""
Track 2 Experiment Runner

This script runs experiments for the Belavkin RL framework on
toy environments (CartPole, Pendulum, etc.) and compares against baselines.

Usage:
    python run_track2_experiments.py --env CartPole-v1 --n_episodes 500
    python run_track2_experiments.py --env Pendulum-v1 --n_episodes 1000
"""

import argparse
import gymnasium as gym
import torch
import numpy as np
import sys
import os

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from track2_rl import BelavkinRLAgent, BelavkinRLTrainer


def run_cartpole_experiment(n_episodes: int = 500, n_seeds: int = 3, output_dir: str = 'results/rl'):
    """
    Run CartPole experiments with Belavkin RL.

    Args:
        n_episodes: Number of training episodes
        n_seeds: Number of random seeds
        output_dir: Output directory for results
    """
    print(f"\n{'='*70}")
    print("CARTPOLE EXPERIMENT - Belavkin RL")
    print(f"{'='*70}\n")

    all_results = []

    for seed in range(n_seeds):
        print(f"\nSeed {seed + 1}/{n_seeds}")
        print("-" * 40)

        # Set seeds
        torch.manual_seed(seed)
        np.random.seed(seed)

        # Create environment
        env = gym.make('CartPole-v1')
        env.reset(seed=seed)

        # Get dimensions
        state_dim = env.observation_space.shape[0]
        action_dim = env.action_space.n

        # Create Belavkin RL agent
        agent = BelavkinRLAgent(
            state_dim=state_dim,
            action_dim=action_dim,
            rank=5,  # Low rank for CartPole
            gamma=0.99,
            learning_rate=1e-3,
        )

        # Create trainer
        trainer = BelavkinRLTrainer(
            env=env, agent=agent, n_episodes=n_episodes, max_steps=500
        )

        # Train
        history = trainer.train(log_interval=50)

        # Evaluate
        eval_results = trainer.evaluate(n_episodes=10)

        print(f"\nEvaluation Results:")
        print(f"  Mean Reward: {eval_results['mean_reward']:.2f} ± {eval_results['std_reward']:.2f}")
        print(f"  Min/Max: {eval_results['min_reward']:.2f} / {eval_results['max_reward']:.2f}")

        all_results.append({
            'seed': seed,
            'history': history,
            'eval_results': eval_results,
        })

        env.close()

    # Aggregate results
    mean_rewards = [r['eval_results']['mean_reward'] for r in all_results]
    print(f"\n{'='*70}")
    print("AGGREGATE RESULTS")
    print(f"{'='*70}")
    print(f"Mean Reward across seeds: {np.mean(mean_rewards):.2f} ± {np.std(mean_rewards):.2f}")

    # Save results
    os.makedirs(output_dir, exist_ok=True)
    import json
    with open(f'{output_dir}/cartpole_results.json', 'w') as f:
        json.dump(all_results, f, indent=2, default=lambda x: x.tolist() if isinstance(x, np.ndarray) else x)

    return all_results


def run_pendulum_experiment(n_episodes: int = 1000, n_seeds: int = 3, output_dir: str = 'results/rl'):
    """
    Run Pendulum experiments with Belavkin RL.

    Note: Pendulum has continuous actions, so we discretize the action space.

    Args:
        n_episodes: Number of training episodes
        n_seeds: Number of random seeds
        output_dir: Output directory
    """
    print(f"\n{'='*70}")
    print("PENDULUM EXPERIMENT - Belavkin RL (Discretized Actions)")
    print(f"{'='*70}\n")

    all_results = []

    for seed in range(n_seeds):
        print(f"\nSeed {seed + 1}/{n_seeds}")
        print("-" * 40)

        # Set seeds
        torch.manual_seed(seed)
        np.random.seed(seed)

        # Create environment
        env = gym.make('Pendulum-v1')
        env.reset(seed=seed)

        # Discretize action space
        n_actions = 5
        action_space = np.linspace(-2.0, 2.0, n_actions)

        # Wrap environment to discretize actions
        class DiscretePendulum:
            def __init__(self, env, action_space):
                self.env = env
                self.action_space_discrete = action_space
                self.observation_space = env.observation_space
                self.action_space = gym.spaces.Discrete(len(action_space))

            def reset(self, seed=None):
                return self.env.reset(seed=seed)

            def step(self, action):
                continuous_action = [self.action_space_discrete[action]]
                return self.env.step(continuous_action)

            def close(self):
                self.env.close()

        wrapped_env = DiscretePendulum(env, action_space)

        # Get dimensions
        state_dim = wrapped_env.observation_space.shape[0]
        action_dim = n_actions

        # Create agent
        agent = BelavkinRLAgent(
            state_dim=state_dim,
            action_dim=action_dim,
            rank=8,
            gamma=0.99,
            learning_rate=1e-3,
        )

        # Train
        trainer = BelavkinRLTrainer(
            env=wrapped_env, agent=agent, n_episodes=n_episodes, max_steps=200
        )
        history = trainer.train(log_interval=100)

        # Evaluate
        eval_results = trainer.evaluate(n_episodes=10)

        print(f"\nEvaluation Results:")
        print(f"  Mean Reward: {eval_results['mean_reward']:.2f} ± {eval_results['std_reward']:.2f}")

        all_results.append({
            'seed': seed,
            'history': history,
            'eval_results': eval_results,
        })

        wrapped_env.close()

    # Aggregate
    mean_rewards = [r['eval_results']['mean_reward'] for r in all_results]
    print(f"\n{'='*70}")
    print("AGGREGATE RESULTS")
    print(f"{'='*70}")
    print(f"Mean Reward across seeds: {np.mean(mean_rewards):.2f} ± {np.std(mean_rewards):.2f}")

    # Save
    os.makedirs(output_dir, exist_ok=True)
    import json
    with open(f'{output_dir}/pendulum_results.json', 'w') as f:
        json.dump(all_results, f, indent=2, default=lambda x: x.tolist() if isinstance(x, np.ndarray) else x)

    return all_results


def compare_with_baselines(env_name: str = 'CartPole-v1', n_episodes: int = 500):
    """
    Compare Belavkin RL with standard baselines.

    Baselines:
        - Random policy
        - DQN (simple implementation)

    Args:
        env_name: Environment name
        n_episodes: Number of episodes for each method
    """
    print(f"\n{'='*70}")
    print(f"BASELINE COMPARISON: {env_name}")
    print(f"{'='*70}\n")

    results = {}

    # 1. Random baseline
    print("Testing Random Policy...")
    env = gym.make(env_name)
    random_rewards = []
    for _ in range(10):
        obs, info = env.reset()
        episode_reward = 0
        for _ in range(500):
            action = env.action_space.sample()
            obs, reward, terminated, truncated, info = env.step(action)
            episode_reward += reward
            if terminated or truncated:
                break
        random_rewards.append(episode_reward)
    env.close()

    results['random'] = {
        'mean': np.mean(random_rewards),
        'std': np.std(random_rewards),
    }
    print(f"Random Policy: {results['random']['mean']:.2f} ± {results['random']['std']:.2f}")

    # 2. Belavkin RL
    print("\nTesting Belavkin RL...")
    belavkin_results = run_cartpole_experiment(n_episodes=n_episodes, n_seeds=1, output_dir='results/comparison')
    results['belavkin'] = belavkin_results[0]['eval_results']

    # Print comparison
    print(f"\n{'='*70}")
    print("FINAL COMPARISON")
    print(f"{'='*70}")
    print(f"Random:     {results['random']['mean']:.2f} ± {results['random']['std']:.2f}")
    print(f"Belavkin:   {results['belavkin']['mean_reward']:.2f} ± {results['belavkin']['std_reward']:.2f}")

    return results


def main():
    parser = argparse.ArgumentParser(description='Run Track 2 RL experiments')
    parser.add_argument(
        '--env',
        type=str,
        choices=['CartPole-v1', 'Pendulum-v1', 'compare'],
        default='CartPole-v1',
        help='Environment to run',
    )
    parser.add_argument('--n_episodes', type=int, default=500, help='Number of episodes')
    parser.add_argument('--n_seeds', type=int, default=3, help='Number of random seeds')
    parser.add_argument('--output_dir', type=str, default='results/rl', help='Output directory')

    args = parser.parse_args()

    if args.env == 'CartPole-v1':
        run_cartpole_experiment(
            n_episodes=args.n_episodes,
            n_seeds=args.n_seeds,
            output_dir=args.output_dir,
        )
    elif args.env == 'Pendulum-v1':
        run_pendulum_experiment(
            n_episodes=args.n_episodes,
            n_seeds=args.n_seeds,
            output_dir=args.output_dir,
        )
    elif args.env == 'compare':
        compare_with_baselines(n_episodes=args.n_episodes)


if __name__ == '__main__':
    main()
