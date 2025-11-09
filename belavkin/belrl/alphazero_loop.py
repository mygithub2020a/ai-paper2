import torch
import torch.optim as optim
import numpy as np
import chess
from belavkin.belrl.game import ChessGame
from belavkin.belrl.tictactoe import TicTacToeGame
from belavkin.belrl.models import PolicyValueNet
from belavkin.belrl.mcts import MCTS
from belavkin.belopt.optim import BelOpt

class ReplayBuffer:
    def __init__(self, capacity):
        self.capacity = capacity
        self.buffer = []

    def push(self, data):
        if len(self.buffer) == self.capacity:
            self.buffer.pop(0)
        self.buffer.append(data)

    def sample(self, batch_size):
        indices = np.random.choice(len(self.buffer), batch_size, replace=False)
        return [self.buffer[i] for i in indices]

    def __len__(self_):
        return len(self_.buffer)

class AlphaZero:
    def __init__(self, game, model, optimizer, replay_buffer, args):
        self.game = game
        self.model = model
        self.optimizer = optimizer
        self.replay_buffer = replay_buffer
        self.args = args
        self.mcts = MCTS(game, model, args)

    def self_play(self):
        """Generates training data through self-play."""
        history = []
        if isinstance(self.game, ChessGame):
            self.game.board = chess.Board()
        else:
            self.game.board = np.zeros((3, 3), dtype=int)

        player = 1

        while not self.game.is_game_over(self.game.board):
            # Get action probabilities from MCTS
            action_probs = self.mcts.get_action_prob(self.game.board, player, temp=self.args.temperature)

            # Choose an action
            action = np.random.choice(len(action_probs), p=action_probs)

            # Store the state, action probabilities, and current player
            history.append([self.game.get_canonical_form(self.game.board, player), action_probs, player])

            # Make the move
            if isinstance(self.game, TicTacToeGame):
                row, col = action // 3, action % 3
                if self.game.board[row, col] == 0:
                    self.game.board[row, col] = player
            else:
                self.game.make_move(action)

            player *= -1

        # Update the history with the game result
        result = self.game.get_game_result(self.game.board)
        for i, data in enumerate(history):
            if data[2] == result:
                data[2] = 1
            else:
                data[2] = -1

        return history

    def train_step(self, batch):
        """Performs a single training step."""
        states, target_pis, target_vs = zip(*batch)

        states = torch.tensor(np.array(states), dtype=torch.float32)
        target_pis = torch.tensor(np.array(target_pis), dtype=torch.float32)
        target_vs = torch.tensor(np.array(target_vs), dtype=torch.float32).unsqueeze(1)

        # Forward pass
        log_pis, vs = self.model(states)

        # Calculate loss
        loss_pi = -torch.sum(target_pis * log_pis) / len(target_pis)
        loss_v = torch.mean((target_vs - vs)**2)
        total_loss = loss_pi + loss_v

        # Backward pass and optimization
        self.optimizer.zero_grad()
        total_loss.backward()
        self.optimizer.step()

    def run(self):
        """The main training loop."""
        for i in range(self.args.num_episodes):
            print(f"Episode {i+1}/{self.args.num_episodes}")
            # 1. Self-play to generate data
            training_data = self.self_play()
            for data in training_data:
                self.replay_buffer.push(data)

            # 2. Sample from replay buffer and train
            if len(self.replay_buffer) > self.args.batch_size:
                print("Training...")
                for _ in range(self.args.num_train_steps):
                    batch = self.replay_buffer.sample(self.args.batch_size)
                    self.train_step(batch)
