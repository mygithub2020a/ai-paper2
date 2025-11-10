"""
Example experiment: Belavkin RL on noisy gridworld.

This script demonstrates:
1. Creating a noisy gridworld environment
2. Training Belavkin DQN agent
3. Comparing with baseline RL algorithms
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import torch
import numpy as np
from belavkin_ml.rl.environments import NoisyGridWorld
from belavkin_ml.rl.agents import BelavkinDQN
from belavkin_ml.benchmarks.rl_bench import RLBenchmark, run_rl_comparison
import matplotlib.pyplot as plt


def main():
    """Run noisy gridworld experiment."""
    print("="*60)
    print("Belavkin RL - Noisy Gridworld")
    print("="*60)

    # Configuration
    grid_size = 5
    noise_prob = 0.2
    obs_noise_std = 0.5
    max_episodes = 500
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    print(f"\nConfiguration:")
    print(f"  Grid size: {grid_size}x{grid_size}")
    print(f"  Noise prob: {noise_prob}")
    print(f"  Obs noise std: {obs_noise_std}")
    print(f"  Max episodes: {max_episodes}")
    print(f"  Device: {device}")

    # Environment factory
    def env_factory():
        return NoisyGridWorld(
            grid_size=grid_size,
            noise_prob=noise_prob,
            obs_noise_std=obs_noise_std,
            sparse_reward=True,
        )

    # Create benchmark
    benchmark = RLBenchmark(
        env_factory=env_factory,
        max_episodes=max_episodes,
        max_steps_per_episode=50,
        eval_every=25,
        eval_episodes=10,
        device=device,
    )

    # Define agents
    state_dim = grid_size * grid_size
    obs_dim = 2
    action_dim = 4

    agents = {
        'BelavkinDQN': lambda: BelavkinDQN(
            state_dim=state_dim,
            obs_dim=obs_dim,
            action_dim=action_dim,
            hidden_dims=[64, 64],
            lr=1e-3,
            gamma=0.99,
            epsilon=0.1,
            device=device,
        ),
        'BelavkinDQN-LowRank': lambda: BelavkinDQN(
            state_dim=state_dim,
            obs_dim=obs_dim,
            action_dim=action_dim,
            hidden_dims=[64, 64],
            lr=1e-3,
            gamma=0.99,
            epsilon=0.1,
            use_low_rank=True,
            n_particles=50,
            device=device,
        ),
    }

    # Run comparison
    results_dir = Path(__file__).parent / 'results' / 'gridworld'
    results_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*60}")
    print("Training agents...")
    print(f"{'='*60}")

    all_results = run_rl_comparison(
        benchmark=benchmark,
        agents=agents,
        n_seeds=3,
        save_dir=results_dir,
        verbose=True,
    )

    # Plot results
    print(f"\nGenerating plots...")

    fig, ax = plt.subplots(figsize=(12, 6))

    for agent_name, agent_results in all_results.items():
        # Average over seeds
        eval_episodes = agent_results[0]['eval_episodes']
        eval_rewards_all = [r['eval_rewards'] for r in agent_results]

        # Compute mean and std
        mean_rewards = np.mean(eval_rewards_all, axis=0)
        std_rewards = np.std(eval_rewards_all, axis=0)

        ax.plot(eval_episodes, mean_rewards, label=agent_name, linewidth=2)
        ax.fill_between(
            eval_episodes,
            mean_rewards - std_rewards,
            mean_rewards + std_rewards,
            alpha=0.2,
        )

    ax.set_xlabel('Episode')
    ax.set_ylabel('Average Reward')
    ax.set_title(f'Noisy Gridworld ({grid_size}x{grid_size})')
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(results_dir / 'learning_curves.png', dpi=300, bbox_inches='tight')
    plt.close()

    print(f"\nResults saved to: {results_dir}")

    # Print summary
    print(f"\n{'='*60}")
    print("Summary")
    print(f"{'='*60}")

    for agent_name, agent_results in all_results.items():
        final_rewards = [r['eval_rewards'][-1] for r in agent_results]
        print(f"\n{agent_name}:")
        print(f"  Final reward: {np.mean(final_rewards):.3f} ± {np.std(final_rewards):.3f}")


if __name__ == '__main__':
    main()
