"""
Example experiment: Belavkin RL on Noisy Gridworld

Demonstrates both model-based and model-free Belavkin agents on a
partially observable gridworld environment.

Usage:
    python run_gridworld.py --agent model_based --episodes 1000
    python run_gridworld.py --agent model_free --episodes 1000
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import argparse
from tqdm import tqdm
import json

from belavkin_ml.rl.environments import NoisyGridworld
from belavkin_ml.rl.agents import ModelBasedBelavkinAgent, ModelFreeBelavkinAgent
from belavkin_ml.rl.belief_state import BeliefStateConfig

sns.set_style('whitegrid')


def train_model_based(env, agent, n_episodes: int, max_steps: int = 50):
    """Train model-based agent."""
    episode_returns = []
    episode_lengths = []
    model_losses = []

    for episode in tqdm(range(n_episodes), desc="Training Model-Based Agent"):
        state, observation = env.reset()
        agent.reset()

        episode_return = 0
        episode_length = 0

        for step in range(max_steps):
            # Select action
            action = agent.select_action(observation, explore=True)

            # Take step
            next_state, next_observation, reward, done, info = env.step(action.item())

            # Update belief state
            agent.update_belief(action, next_observation, reward)

            # Store transition
            agent.store_transition(state, action, reward, next_state, done)

            # Train models periodically
            if len(agent.buffer) >= 64:
                losses = agent.train_models(batch_size=64, n_epochs=1)
                if losses:
                    model_losses.append(losses)

                # Train policy periodically
                if step % 10 == 0:
                    agent.train_policy(batch_size=64)

            episode_return += reward
            episode_length += 1

            state = next_state
            observation = next_observation

            if done:
                break

        episode_returns.append(episode_return)
        episode_lengths.append(episode_length)

    return {
        'episode_returns': episode_returns,
        'episode_lengths': episode_lengths,
        'model_losses': model_losses,
    }


def train_model_free(env, agent, n_episodes: int, max_steps: int = 50):
    """Train model-free agent."""
    episode_returns = []
    episode_lengths = []
    q_losses = []

    # Epsilon decay schedule
    epsilon_start = 1.0
    epsilon_end = 0.01
    epsilon_decay = 0.995

    epsilon = epsilon_start

    for episode in tqdm(range(n_episodes), desc="Training Model-Free Agent"):
        state, observation = env.reset()

        episode_return = 0
        episode_length = 0

        for step in range(max_steps):
            # Select action (use observation as state for model-free)
            action = agent.select_action(observation, epsilon=epsilon)

            # Take step
            next_state, next_observation, reward, done, info = env.step(action.item())

            # Store transition (use observations)
            agent.store_transition(observation, action, reward, next_observation, done)

            # Train Q-network
            if len(agent.buffer) >= 64:
                losses = agent.train(batch_size=64)
                if losses:
                    q_losses.append(losses['q_loss'])

                # Update target network periodically
                if episode % 10 == 0 and step == 0:
                    agent.update_target_network()

            episode_return += reward
            episode_length += 1

            state = next_state
            observation = next_observation

            if done:
                break

        # Decay epsilon
        epsilon = max(epsilon_end, epsilon * epsilon_decay)

        episode_returns.append(episode_return)
        episode_lengths.append(episode_length)

    return {
        'episode_returns': episode_returns,
        'episode_lengths': episode_lengths,
        'q_losses': q_losses,
    }


def evaluate_agent(env, agent, n_episodes: int = 100, max_steps: int = 50, model_based: bool = True):
    """Evaluate trained agent."""
    episode_returns = []
    episode_lengths = []
    success_count = 0

    for episode in range(n_episodes):
        state, observation = env.reset()

        if model_based:
            agent.reset()

        episode_return = 0
        episode_length = 0
        reached_goal = False

        for step in range(max_steps):
            # Select action (no exploration)
            if model_based:
                action = agent.select_action(observation, explore=False)
                agent.update_belief(action, observation, 0.0)
            else:
                action = agent.select_action(observation, epsilon=0.0)

            # Take step
            next_state, next_observation, reward, done, info = env.step(action.item())

            episode_return += reward
            episode_length += 1

            observation = next_observation

            if done:
                reached_goal = True
                break

        if reached_goal:
            success_count += 1

        episode_returns.append(episode_return)
        episode_lengths.append(episode_length)

    return {
        'mean_return': np.mean(episode_returns),
        'std_return': np.std(episode_returns),
        'mean_length': np.mean(episode_lengths),
        'success_rate': success_count / n_episodes,
    }


def plot_results(results, save_path: Path):
    """Plot training results."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Episode returns
    returns = results['episode_returns']
    window = 50
    smoothed = np.convolve(returns, np.ones(window)/window, mode='valid')

    axes[0].plot(returns, alpha=0.3, label='Raw')
    axes[0].plot(range(window-1, len(returns)), smoothed, linewidth=2, label=f'{window}-episode MA')
    axes[0].set_xlabel('Episode')
    axes[0].set_ylabel('Return')
    axes[0].set_title('Training Returns')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # Episode lengths
    lengths = results['episode_lengths']
    smoothed_lengths = np.convolve(lengths, np.ones(window)/window, mode='valid')

    axes[1].plot(lengths, alpha=0.3, label='Raw')
    axes[1].plot(range(window-1, len(lengths)), smoothed_lengths, linewidth=2, label=f'{window}-episode MA')
    axes[1].set_xlabel('Episode')
    axes[1].set_ylabel('Episode Length')
    axes[1].set_title('Episode Lengths')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Saved plot to {save_path}")


def main(args):
    print("="*70)
    print("Belavkin RL: Noisy Gridworld Experiment")
    print("="*70)

    # Set device
    device = 'cuda' if torch.cuda.is_available() and not args.cpu else 'cpu'
    print(f"Device: {device}")

    # Set seed
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # Create environment
    env = NoisyGridworld(
        size=args.grid_size,
        observation_noise=args.obs_noise,
        goal_reward=10.0,
        step_penalty=-0.1,
    )

    print(f"\nEnvironment:")
    print(f"  Grid size: {args.grid_size}x{args.grid_size}")
    print(f"  Observation noise: {args.obs_noise}")
    print(f"  State dim: {env.state_dim}")
    print(f"  Action dim: {env.action_dim}")

    # Create agent
    if args.agent == 'model_based':
        print(f"\nCreating Model-Based Belavkin Agent...")

        belief_config = BeliefStateConfig(
            state_dim=env.state_dim,
            action_dim=env.action_dim,
            observation_dim=env.observation_dim,
            representation=args.belief_repr,
            rank=args.belief_rank,
            measurement_strength=0.1,
        )

        agent = ModelBasedBelavkinAgent(
            state_dim=env.state_dim,
            action_dim=env.action_dim,
            observation_dim=env.observation_dim,
            belief_config=belief_config,
            lr=args.lr,
            gamma=args.gamma,
            device=device,
            discrete_actions=True,
        )

        # Train
        print(f"\nTraining for {args.episodes} episodes...")
        results = train_model_based(env, agent, args.episodes, args.max_steps)

    elif args.agent == 'model_free':
        print(f"\nCreating Model-Free Belavkin Agent...")

        agent = ModelFreeBelavkinAgent(
            state_dim=env.observation_dim,
            action_dim=env.action_dim,
            lr=args.lr,
            gamma=args.gamma,
            damping=args.damping,
            exploration=args.beta,
            device=device,
            discrete_actions=True,
        )

        # Train
        print(f"\nTraining for {args.episodes} episodes...")
        results = train_model_free(env, agent, args.episodes, args.max_steps)

    else:
        raise ValueError(f"Unknown agent type: {args.agent}")

    # Evaluate
    print("\nEvaluating...")
    eval_results = evaluate_agent(
        env, agent, n_episodes=100, max_steps=args.max_steps,
        model_based=(args.agent == 'model_based')
    )

    print("\nEvaluation Results:")
    print(f"  Mean return: {eval_results['mean_return']:.2f} ± {eval_results['std_return']:.2f}")
    print(f"  Mean episode length: {eval_results['mean_length']:.2f}")
    print(f"  Success rate: {eval_results['success_rate']:.1%}")

    # Save results
    save_dir = Path(args.save_dir) / f"{args.agent}_{args.grid_size}x{args.grid_size}"
    save_dir.mkdir(parents=True, exist_ok=True)

    # Plot
    plot_results(results, save_dir / "training_curves.png")

    # Save data
    results['eval'] = eval_results
    results['config'] = vars(args)

    # Convert numpy arrays to lists for JSON serialization
    json_results = {}
    for key, value in results.items():
        if isinstance(value, list) and len(value) > 0:
            if isinstance(value[0], dict):
                json_results[key] = value
            else:
                json_results[key] = [float(x) if not isinstance(x, dict) else x for x in value]
        else:
            json_results[key] = value

    with open(save_dir / "results.json", 'w') as f:
        json.dump(json_results, f, indent=2)

    print(f"\nResults saved to {save_dir}")
    print("="*70)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Belavkin RL on Noisy Gridworld")

    # Environment
    parser.add_argument("--grid_size", type=int, default=5, help="Grid size")
    parser.add_argument("--obs_noise", type=float, default=0.5, help="Observation noise")

    # Agent
    parser.add_argument("--agent", type=str, default="model_based",
                        choices=["model_based", "model_free"],
                        help="Agent type")

    # Belief state (model-based only)
    parser.add_argument("--belief_repr", type=str, default="low_rank",
                        choices=["low_rank", "neural", "particle"],
                        help="Belief state representation")
    parser.add_argument("--belief_rank", type=int, default=10,
                        help="Rank for low-rank belief state")

    # Training
    parser.add_argument("--episodes", type=int, default=1000, help="Number of episodes")
    parser.add_argument("--max_steps", type=int, default=50, help="Max steps per episode")
    parser.add_argument("--lr", type=float, default=3e-4, help="Learning rate")
    parser.add_argument("--gamma", type=float, default=0.99, help="Discount factor")

    # Belavkin parameters (model-free only)
    parser.add_argument("--damping", type=float, default=1e-4, help="Damping factor γ")
    parser.add_argument("--beta", type=float, default=1e-2, help="Exploration factor β")

    # Other
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--cpu", action="store_true", help="Force CPU usage")
    parser.add_argument("--save_dir", type=str, default="experiments/track2_rl/results",
                        help="Save directory")

    args = parser.parse_args()
    main(args)
