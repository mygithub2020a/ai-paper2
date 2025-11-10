"""
BelRL Trainer: AlphaZero-style training with BelOpt optimizer.
"""

import os
import time
from typing import Optional, List, Tuple, Dict
from collections import deque
import numpy as np
import torch
import torch.nn as nn

from .mcts import MCTS, MCTSConfig
from .models import PolicyValueNetwork
from ..belopt import BelOpt


class ReplayBuffer:
    """Experience replay buffer for storing self-play games."""

    def __init__(self, max_size: int = 100000):
        self.buffer = deque(maxlen=max_size)

    def add(self, state, policy, value):
        """Add experience to buffer."""
        self.buffer.append((state, policy, value))

    def sample(self, batch_size: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Sample batch from buffer."""
        indices = np.random.choice(len(self.buffer), batch_size, replace=False)
        batch = [self.buffer[i] for i in indices]

        states = torch.stack([x[0] for x in batch])
        policies = torch.stack([x[1] for x in batch])
        values = torch.tensor([x[2] for x in batch], dtype=torch.float32).unsqueeze(1)

        return states, policies, values

    def __len__(self):
        return len(self.buffer)


class BelRLTrainer:
    """
    BelRL Trainer for AlphaZero-style reinforcement learning.

    Uses BelOpt optimizer with MCTS-guided self-play.
    """

    def __init__(
        self,
        network: nn.Module,
        optimizer_name: str = 'belopt',
        lr: float = 1e-3,
        gamma0: float = 1e-3,
        beta0: float = 1e-3,
        weight_decay: float = 1e-4,
        mcts_config: Optional[MCTSConfig] = None,
        buffer_size: int = 100000,
        batch_size: int = 256,
        device: str = 'cuda' if torch.cuda.is_available() else 'cpu',
    ):
        """
        Args:
            network: Policy-value network
            optimizer_name: Optimizer to use ('belopt', 'adam', 'sgd')
            lr: Learning rate
            gamma0: BelOpt gamma parameter
            beta0: BelOpt beta parameter (exploration noise)
            weight_decay: Weight decay
            mcts_config: MCTS configuration
            buffer_size: Replay buffer size
            batch_size: Training batch size
            device: Device to use
        """
        self.network = network.to(device)
        self.device = device
        self.batch_size = batch_size

        # Create optimizer
        if optimizer_name.lower() == 'belopt':
            self.optimizer = BelOpt(
                network.parameters(),
                lr=lr,
                gamma0=gamma0,
                beta0=beta0,
                weight_decay=weight_decay,
                decoupled_weight_decay=True,
            )
        elif optimizer_name.lower() == 'adam':
            self.optimizer = torch.optim.Adam(
                network.parameters(),
                lr=lr,
                weight_decay=weight_decay
            )
        elif optimizer_name.lower() == 'sgd':
            self.optimizer = torch.optim.SGD(
                network.parameters(),
                lr=lr,
                momentum=0.9,
                weight_decay=weight_decay
            )
        else:
            raise ValueError(f"Unknown optimizer: {optimizer_name}")

        # MCTS
        self.mcts_config = mcts_config if mcts_config else MCTSConfig()
        self.mcts = MCTS(self.mcts_config)

        # Replay buffer
        self.replay_buffer = ReplayBuffer(max_size=buffer_size)

        # Training statistics
        self.train_stats = {
            'games_played': 0,
            'training_steps': 0,
            'policy_loss': [],
            'value_loss': [],
            'total_loss': [],
        }

    def self_play(self, game_class, num_games: int = 1) -> List[List[Tuple]]:
        """
        Generate games through self-play.

        Args:
            game_class: Game class (should inherit from GameState)
            num_games: Number of games to play

        Returns:
            List of game histories (each history is list of (state, policy, value) tuples)
        """
        self.network.eval()
        games = []

        for _ in range(num_games):
            game_history = []
            state = game_class()

            with torch.no_grad():
                while True:
                    # Run MCTS to get action probabilities
                    action_probs, root = self.mcts.run_simulations(state, self.network)

                    # Store state and policy for training
                    state_tensor = state.to_tensor()
                    policy_tensor = torch.zeros(state.action_size())
                    for action, prob in action_probs.items():
                        policy_tensor[action] = prob

                    # Store for later (value will be filled in after game ends)
                    game_history.append((state_tensor, policy_tensor, None))

                    # Sample action
                    action = self.mcts.select_action(state, self.network, deterministic=False)

                    # Apply action
                    state.apply_action(action)

                    # Check if game is over
                    terminal, outcome = state.is_terminal()
                    if terminal:
                        # Fill in values based on game outcome
                        completed_history = []
                        for i, (s, p, _) in enumerate(game_history):
                            # Value from perspective of player who made the move
                            player = 1 if i % 2 == 0 else -1
                            value = outcome * player
                            completed_history.append((s, p, value))

                        games.append(completed_history)
                        break

            self.train_stats['games_played'] += 1

        return games

    def train_step(self) -> Dict[str, float]:
        """
        Perform one training step.

        Returns:
            Dictionary of losses
        """
        if len(self.replay_buffer) < self.batch_size:
            return {'policy_loss': 0.0, 'value_loss': 0.0, 'total_loss': 0.0}

        self.network.train()

        # Sample batch from replay buffer
        states, target_policies, target_values = self.replay_buffer.sample(self.batch_size)
        states = states.to(self.device)
        target_policies = target_policies.to(self.device)
        target_values = target_values.to(self.device)

        # Forward pass
        pred_policies, pred_values = self.network(states)

        # Compute losses
        policy_loss = -torch.mean(torch.sum(target_policies * pred_policies, dim=1))
        value_loss = nn.MSELoss()(pred_values, target_values)
        total_loss = policy_loss + value_loss

        # Backward pass
        self.optimizer.zero_grad()
        total_loss.backward()
        self.optimizer.step()

        # Record statistics
        self.train_stats['training_steps'] += 1
        losses = {
            'policy_loss': policy_loss.item(),
            'value_loss': value_loss.item(),
            'total_loss': total_loss.item(),
        }

        for key, value in losses.items():
            self.train_stats[key].append(value)

        return losses

    def train(
        self,
        game_class,
        num_iterations: int,
        games_per_iteration: int = 100,
        train_steps_per_iteration: int = 1000,
        save_dir: Optional[str] = None,
    ):
        """
        Main training loop.

        Args:
            game_class: Game class
            num_iterations: Number of training iterations
            games_per_iteration: Self-play games per iteration
            train_steps_per_iteration: Training steps per iteration
            save_dir: Directory to save checkpoints
        """
        if save_dir:
            os.makedirs(save_dir, exist_ok=True)

        for iteration in range(num_iterations):
            print(f"\nIteration {iteration + 1}/{num_iterations}")

            # Self-play
            print(f"  Generating {games_per_iteration} self-play games...")
            start_time = time.time()
            games = self.self_play(game_class, num_games=games_per_iteration)

            # Add to replay buffer
            for game_history in games:
                for state, policy, value in game_history:
                    self.replay_buffer.add(state, policy, value)

            self_play_time = time.time() - start_time
            print(f"    Self-play time: {self_play_time:.2f}s")
            print(f"    Buffer size: {len(self.replay_buffer)}")

            # Training
            print(f"  Training for {train_steps_per_iteration} steps...")
            start_time = time.time()
            total_policy_loss = 0.0
            total_value_loss = 0.0

            for step in range(train_steps_per_iteration):
                losses = self.train_step()
                total_policy_loss += losses['policy_loss']
                total_value_loss += losses['value_loss']

                if (step + 1) % 100 == 0:
                    avg_policy = total_policy_loss / (step + 1)
                    avg_value = total_value_loss / (step + 1)
                    print(f"    Step {step + 1}: Policy Loss: {avg_policy:.4f}, "
                          f"Value Loss: {avg_value:.4f}")

            train_time = time.time() - start_time
            print(f"    Training time: {train_time:.2f}s")

            # Save checkpoint
            if save_dir and (iteration + 1) % 10 == 0:
                checkpoint_path = os.path.join(save_dir, f"checkpoint_{iteration + 1}.pt")
                self.save_checkpoint(checkpoint_path)
                print(f"    Saved checkpoint: {checkpoint_path}")

    def save_checkpoint(self, path: str):
        """Save training checkpoint."""
        torch.save({
            'network_state_dict': self.network.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'train_stats': self.train_stats,
        }, path)

    def load_checkpoint(self, path: str):
        """Load training checkpoint."""
        checkpoint = torch.load(path)
        self.network.load_state_dict(checkpoint['network_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.train_stats = checkpoint['train_stats']

    def evaluate(self, game_class, opponent, num_games: int = 100) -> Dict[str, float]:
        """
        Evaluate against an opponent.

        Args:
            game_class: Game class
            opponent: Opponent network or policy
            num_games: Number of evaluation games

        Returns:
            Statistics: win_rate, draw_rate, loss_rate
        """
        self.network.eval()
        wins = draws = losses = 0

        for game_num in range(num_games):
            state = game_class()
            our_color = 1 if game_num % 2 == 0 else -1

            with torch.no_grad():
                while True:
                    if state.to_play() == our_color:
                        # Our move
                        action = self.mcts.select_action(state, self.network, deterministic=True)
                    else:
                        # Opponent's move
                        if hasattr(opponent, 'select_action'):
                            action = opponent.select_action(state)
                        else:
                            # Use MCTS with opponent network
                            mcts_opp = MCTS(self.mcts_config)
                            action = mcts_opp.select_action(state, opponent, deterministic=True)

                    state.apply_action(action)

                    terminal, outcome = state.is_terminal()
                    if terminal:
                        if outcome * our_color > 0:
                            wins += 1
                        elif outcome * our_color < 0:
                            losses += 1
                        else:
                            draws += 1
                        break

        return {
            'win_rate': wins / num_games,
            'draw_rate': draws / num_games,
            'loss_rate': losses / num_games,
        }


if __name__ == '__main__':
    print("BelRL Trainer implementation complete!")
    print("Example usage:")
    print("""
    from belavkin.belrl import PolicyValueNetwork, BelRLTrainer, MCTSConfig

    # Create network
    network = PolicyValueNetwork(board_size=8, action_size=64)

    # Create trainer with BelOpt
    trainer = BelRLTrainer(
        network=network,
        optimizer_name='belopt',
        lr=1e-3,
        gamma0=1e-3,
        beta0=1e-3,
    )

    # Train
    trainer.train(YourGameClass, num_iterations=100)
    """)
