"""
Train Belavkin RL agents on toy environments.

This script demonstrates how to train both model-based and model-free
Belavkin RL agents on partially observable environments.

Usage:
    python experiments/train_belavkin_rl.py --env gridworld --agent model-free
    python experiments/train_belavkin_rl.py --env pendulum --agent model-based
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import numpy as np
import argparse
from tqdm import tqdm

from track2_rl.model_free import ModelFreeBelavkinRL
from track2_rl.model_based import ModelBasedBelavkinRL
from track2_rl.environments.noisy_gridworld import NoisyGridworld
from track2_rl.environments.pendulum_partial import PartialObservabilityPendulum
from track2_rl.environments.utils import ReplayBuffer, collect_episode, evaluate_agent


def train_rl_agent(
    agent_type: str = "model-free",
    env_name: str = "gridworld",
    num_episodes: int = 500,
    batch_size: int = 64,
    buffer_size: int = 10000,
    learning_starts: int = 100,
    eval_freq: int = 50,
    save_dir: str = "./results/track2_rl",
):
    """
    Train a Belavkin RL agent.

    Args:
        agent_type: "model-free" or "model-based"
        env_name: "gridworld" or "pendulum"
        num_episodes: Number of training episodes
        batch_size: Batch size for training
        buffer_size: Replay buffer capacity
        learning_starts: Start training after N episodes
        eval_freq: Evaluate every N episodes
        save_dir: Directory to save results
    """
    print("="*60)
    print(f"Training Belavkin RL Agent")
    print(f"  Agent: {agent_type}")
    print(f"  Environment: {env_name}")
    print("="*60)

    # Create environment
    if env_name == "gridworld":
        env = NoisyGridworld(grid_size=10, observation_noise=0.3)
    elif env_name == "pendulum":
        env = PartialObservabilityPendulum(observation_noise=0.1)
    else:
        raise ValueError(f"Unknown environment: {env_name}")

    print(f"\nEnvironment Details:")
    print(f"  State dim: {env.state_dim}")
    print(f"  Action dim: {env.action_dim}")
    print(f"  Observation dim: {env.observation_dim}")

    # Create agent
    if agent_type == "model-free":
        agent = ModelFreeBelavkinRL(
            state_dim=env.observation_dim,
            action_dim=env.action_dim,
            belief_dim=32,
            hidden_dim=128,
            lr=3e-4,
        )
    elif agent_type == "model-based":
        agent = ModelBasedBelavkinRL(
            state_dim=env.observation_dim,
            action_dim=env.action_dim,
            belief_dim=32,
            hidden_dim=128,
            lr=3e-4,
        )
    else:
        raise ValueError(f"Unknown agent type: {agent_type}")

    print(f"\nAgent: {agent_type.upper()}")
    print(f"  Belief dim: 32")
    print(f"  Hidden dim: 128")

    # Create replay buffer
    replay_buffer = ReplayBuffer(capacity=buffer_size)

    # Training loop
    episode_rewards = []
    eval_results = []

    print(f"\nStarting training for {num_episodes} episodes...")
    print(f"  Batch size: {batch_size}")
    print(f"  Learning starts: {learning_starts}")
    print(f"  Evaluation frequency: {eval_freq}")

    for episode in range(num_episodes):
        # Collect episode
        agent.reset()
        total_reward, episode_length, transitions = collect_episode(
            env=env,
            agent=agent,
            max_steps=200,
            training=True,
        )

        # Add to replay buffer
        for trans in transitions:
            replay_buffer.push(
                state=trans['state'],
                action=trans['action'],
                reward=trans['reward'],
                next_state=trans['next_state'],
                done=trans['done'],
            )

        episode_rewards.append(total_reward)

        # Train agent
        if episode >= learning_starts and len(replay_buffer) >= batch_size:
            # Train for multiple steps per episode
            num_train_steps = max(1, episode_length // 4)

            for _ in range(num_train_steps):
                batch = replay_buffer.sample(batch_size)
                metrics = agent.train_step(batch)

        # Logging
        if episode % 10 == 0:
            recent_rewards = episode_rewards[-10:]
            avg_reward = np.mean(recent_rewards)
            print(f"Episode {episode}/{num_episodes} | "
                  f"Reward: {total_reward:.2f} | "
                  f"Avg (last 10): {avg_reward:.2f} | "
                  f"Buffer: {len(replay_buffer)}")

        # Evaluation
        if (episode + 1) % eval_freq == 0 and episode >= learning_starts:
            eval_metrics = evaluate_agent(
                env=env,
                agent=agent,
                num_episodes=10,
                max_steps=200,
            )
            eval_results.append((episode, eval_metrics))

            print(f"\n{'='*50}")
            print(f"Evaluation at episode {episode + 1}")
            print(f"  Mean reward: {eval_metrics['mean_reward']:.2f} ± {eval_metrics['std_reward']:.2f}")
            print(f"  Mean length: {eval_metrics['mean_length']:.1f}")
            print(f"  Min/Max reward: {eval_metrics['min_reward']:.2f} / {eval_metrics['max_reward']:.2f}")
            print(f"{'='*50}\n")

    # Final evaluation
    print("\n" + "="*60)
    print("Final Evaluation")
    print("="*60)

    final_metrics = evaluate_agent(
        env=env,
        agent=agent,
        num_episodes=20,
        max_steps=200,
    )

    print(f"\nFinal Performance (20 episodes):")
    print(f"  Mean reward: {final_metrics['mean_reward']:.2f} ± {final_metrics['std_reward']:.2f}")
    print(f"  Mean length: {final_metrics['mean_length']:.1f}")
    print(f"  Range: [{final_metrics['min_reward']:.2f}, {final_metrics['max_reward']:.2f}]")

    # Save results
    output_path = Path(save_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    results_file = output_path / f"{agent_type}_{env_name}_results.npz"
    np.savez(
        results_file,
        episode_rewards=episode_rewards,
        eval_results=eval_results,
        final_metrics=final_metrics,
    )

    print(f"\nResults saved to: {results_file}")

    return agent, episode_rewards, eval_results


def main():
    parser = argparse.ArgumentParser(description="Train Belavkin RL agents")
    parser.add_argument("--agent", type=str, default="model-free",
                        choices=["model-free", "model-based"],
                        help="Agent type")
    parser.add_argument("--env", type=str, default="gridworld",
                        choices=["gridworld", "pendulum"],
                        help="Environment name")
    parser.add_argument("--episodes", type=int, default=500,
                        help="Number of training episodes")
    parser.add_argument("--batch-size", type=int, default=64,
                        help="Batch size")
    parser.add_argument("--buffer-size", type=int, default=10000,
                        help="Replay buffer size")
    parser.add_argument("--learning-starts", type=int, default=100,
                        help="Start training after N episodes")
    parser.add_argument("--eval-freq", type=int, default=50,
                        help="Evaluation frequency")
    parser.add_argument("--save-dir", type=str, default="./results/track2_rl",
                        help="Directory to save results")

    args = parser.parse_args()

    train_rl_agent(
        agent_type=args.agent,
        env_name=args.env,
        num_episodes=args.episodes,
        batch_size=args.batch_size,
        buffer_size=args.buffer_size,
        learning_starts=args.learning_starts,
        eval_freq=args.eval_freq,
        save_dir=args.save_dir,
    )


if __name__ == "__main__":
    main()
