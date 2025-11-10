"""
Quick RL test to validate Track 2 implementation.
"""

import sys
import os
import torch
import numpy as np
import gymnasium as gym

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from track2_rl import BelavkinRLAgent, BelavkinRLTrainer


def quick_cartpole_test():
    """Run quick CartPole test."""
    print("\n" + "="*70)
    print("QUICK CARTPOLE TEST - BELAVKIN RL")
    print("="*70)
    print("\nSettings: 100 episodes, rank=5")
    print()

    # Set seed
    torch.manual_seed(42)
    np.random.seed(42)

    # Create environment
    env = gym.make('CartPole-v1')
    env.reset(seed=42)

    # Get dimensions
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n

    print(f"State dim: {state_dim}, Action dim: {action_dim}")

    # Create agent
    agent = BelavkinRLAgent(
        state_dim=state_dim,
        action_dim=action_dim,
        rank=5,
        gamma=0.99,
        learning_rate=1e-3
    )

    # Create trainer
    trainer = BelavkinRLTrainer(
        env=env,
        agent=agent,
        n_episodes=100,
        max_steps=500
    )

    # Train
    print("\nTraining...")
    history = trainer.train(log_interval=10)

    # Evaluate
    print("\nEvaluating...")
    results = trainer.evaluate(n_episodes=10)

    print("\n" + "="*70)
    print("RESULTS")
    print("="*70)
    print(f"Mean reward: {results['mean_reward']:.2f} ± {results['std_reward']:.2f}")
    print(f"Min/Max: {results['min_reward']:.2f} / {results['max_reward']:.2f}")

    # Compare with random
    print("\nRandom baseline:")
    random_rewards = []
    for _ in range(10):
        obs, info = env.reset()
        total_reward = 0
        for _ in range(500):
            action = env.action_space.sample()
            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
            if terminated or truncated:
                break
        random_rewards.append(total_reward)

    print(f"Random: {np.mean(random_rewards):.2f} ± {np.std(random_rewards):.2f}")

    env.close()

    print("\n" + "="*70)
    if results['mean_reward'] > np.mean(random_rewards):
        print("✓ BELAVKIN RL OUTPERFORMS RANDOM")
    else:
        print("✗ BELAVKIN RL DOES NOT OUTPERFORM RANDOM")
    print("="*70)

    # Save results
    os.makedirs('results/rl_quick', exist_ok=True)
    import json
    with open('results/rl_quick/cartpole_quick.json', 'w') as f:
        json.dump({
            'belavkin': results,
            'random': {
                'mean_reward': float(np.mean(random_rewards)),
                'std_reward': float(np.std(random_rewards))
            },
            'history': {
                'episode_rewards': [float(x) for x in history['episode_rewards']],
                'episode_lengths': [int(x) for x in history['episode_lengths']]
            }
        }, f, indent=2)

    print("\nResults saved to results/rl_quick/cartpole_quick.json")

    return results


if __name__ == '__main__':
    quick_cartpole_test()
