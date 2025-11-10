"""
Game implementations for BelRL.

Includes simple games for testing and demonstration.
"""

import numpy as np
import torch
from typing import List, Tuple
from .mcts import GameState


class TicTacToe(GameState):
    """Simple Tic-Tac-Toe implementation for testing."""

    def __init__(self):
        self.board = np.zeros((3, 3), dtype=np.int8)
        self.current_player = 1
        self.action_space_size = 9

    def clone(self) -> 'TicTacToe':
        new_game = TicTacToe()
        new_game.board = self.board.copy()
        new_game.current_player = self.current_player
        return new_game

    def apply_action(self, action: int):
        row, col = action // 3, action % 3
        self.board[row, col] = self.current_player
        self.current_player *= -1

    def legal_actions(self) -> List[int]:
        return [i for i in range(9) if self.board[i // 3, i % 3] == 0]

    def is_terminal(self) -> Tuple[bool, float]:
        # Check rows
        for i in range(3):
            if abs(self.board[i, :].sum()) == 3:
                return True, float(np.sign(self.board[i, :].sum()))

        # Check columns
        for j in range(3):
            if abs(self.board[:, j].sum()) == 3:
                return True, float(np.sign(self.board[:, j].sum()))

        # Check diagonals
        if abs(self.board[[0,1,2], [0,1,2]].sum()) == 3:
            return True, float(np.sign(self.board[[0,1,2], [0,1,2]].sum()))
        if abs(self.board[[0,1,2], [2,1,0]].sum()) == 3:
            return True, float(np.sign(self.board[[0,1,2], [2,1,0]].sum()))

        # Check draw
        if len(self.legal_actions()) == 0:
            return True, 0.0

        return False, 0.0

    def to_play(self) -> int:
        return self.current_player

    def to_tensor(self) -> torch.Tensor:
        # Create 2-channel representation: one for each player
        state = np.zeros((2, 3, 3), dtype=np.float32)
        state[0] = (self.board == 1).astype(np.float32)
        state[1] = (self.board == -1).astype(np.float32)
        return torch.from_numpy(state).unsqueeze(0)

    def action_size(self) -> int:
        return self.action_space_size


class ConnectFour(GameState):
    """Connect Four implementation."""

    def __init__(self, rows: int = 6, cols: int = 7):
        self.rows = rows
        self.cols = cols
        self.board = np.zeros((rows, cols), dtype=np.int8)
        self.current_player = 1
        self.last_move = None

    def clone(self) -> 'ConnectFour':
        new_game = ConnectFour(self.rows, self.cols)
        new_game.board = self.board.copy()
        new_game.current_player = self.current_player
        new_game.last_move = self.last_move
        return new_game

    def apply_action(self, action: int):
        """Drop piece in column 'action'."""
        for row in range(self.rows - 1, -1, -1):
            if self.board[row, action] == 0:
                self.board[row, action] = self.current_player
                self.last_move = (row, action)
                self.current_player *= -1
                return
        raise ValueError(f"Column {action} is full!")

    def legal_actions(self) -> List[int]:
        return [col for col in range(self.cols) if self.board[0, col] == 0]

    def is_terminal(self) -> Tuple[bool, float]:
        if self.last_move is None:
            return False, 0.0

        row, col = self.last_move
        player = self.board[row, col]

        # Check horizontal
        count = 1
        for dc in [-1, 1]:
            c = col + dc
            while 0 <= c < self.cols and self.board[row, c] == player:
                count += 1
                c += dc
        if count >= 4:
            return True, float(player)

        # Check vertical
        count = 1
        for dr in [-1, 1]:
            r = row + dr
            while 0 <= r < self.rows and self.board[r, col] == player:
                count += 1
                r += dr
        if count >= 4:
            return True, float(player)

        # Check diagonals
        for dr, dc in [(1, 1), (1, -1)]:
            count = 1
            for direction in [-1, 1]:
                r, c = row + direction * dr, col + direction * dc
                while 0 <= r < self.rows and 0 <= c < self.cols and self.board[r, c] == player:
                    count += 1
                    r += direction * dr
                    c += direction * dc
            if count >= 4:
                return True, float(player)

        # Check draw
        if len(self.legal_actions()) == 0:
            return True, 0.0

        return False, 0.0

    def to_play(self) -> int:
        return self.current_player

    def to_tensor(self) -> torch.Tensor:
        state = np.zeros((2, self.rows, self.cols), dtype=np.float32)
        state[0] = (self.board == 1).astype(np.float32)
        state[1] = (self.board == -1).astype(np.float32)
        return torch.from_numpy(state).unsqueeze(0)

    def action_size(self) -> int:
        return self.cols


class SimpleHex(GameState):
    """
    Simplified Hex game implementation.

    Hex is a connection game where players try to form a connected path
    between their two opposite sides of the board.
    """

    def __init__(self, board_size: int = 11):
        self.board_size = board_size
        self.board = np.zeros((board_size, board_size), dtype=np.int8)
        self.current_player = 1  # Player 1: connect top-bottom, Player 2: connect left-right
        self.move_count = 0

    def clone(self) -> 'SimpleHex':
        new_game = SimpleHex(self.board_size)
        new_game.board = self.board.copy()
        new_game.current_player = self.current_player
        new_game.move_count = self.move_count
        return new_game

    def apply_action(self, action: int):
        row, col = action // self.board_size, action % self.board_size
        if self.board[row, col] != 0:
            raise ValueError(f"Cell ({row}, {col}) is already occupied!")
        self.board[row, col] = self.current_player
        self.move_count += 1
        self.current_player *= -1

    def legal_actions(self) -> List[int]:
        return [
            row * self.board_size + col
            for row in range(self.board_size)
            for col in range(self.board_size)
            if self.board[row, col] == 0
        ]

    def is_terminal(self) -> Tuple[bool, float]:
        # Check if player 1 (connecting top-bottom) has won
        if self._has_won(1):
            return True, 1.0

        # Check if player -1 (connecting left-right) has won
        if self._has_won(-1):
            return True, -1.0

        # No more moves (shouldn't happen in Hex, but check anyway)
        if self.move_count == self.board_size * self.board_size:
            return True, 0.0

        return False, 0.0

    def _has_won(self, player: int) -> bool:
        """Check if player has formed a winning path."""
        visited = set()

        if player == 1:
            # Player 1 connects top to bottom
            # Start from top row
            start_cells = [(0, col) for col in range(self.board_size) if self.board[0, col] == player]
            target_row = self.board_size - 1

            for start in start_cells:
                if self._dfs_path(player, start, visited, lambda r, c: r == target_row):
                    return True

        else:
            # Player -1 connects left to right
            # Start from left column
            start_cells = [(row, 0) for row in range(self.board_size) if self.board[row, 0] == player]
            target_col = self.board_size - 1

            for start in start_cells:
                if self._dfs_path(player, start, visited, lambda r, c: c == target_col):
                    return True

        return False

    def _dfs_path(self, player: int, cell: Tuple[int, int], visited: set, target_condition) -> bool:
        """DFS to find path from start to target."""
        row, col = cell

        if target_condition(row, col):
            return True

        visited.add(cell)

        # Check neighbors (6 directions in hex grid)
        neighbors = [
            (row - 1, col), (row - 1, col + 1),
            (row, col - 1), (row, col + 1),
            (row + 1, col - 1), (row + 1, col)
        ]

        for nr, nc in neighbors:
            if (0 <= nr < self.board_size and 0 <= nc < self.board_size and
                self.board[nr, nc] == player and (nr, nc) not in visited):
                if self._dfs_path(player, (nr, nc), visited, target_condition):
                    return True

        return False

    def to_play(self) -> int:
        return self.current_player

    def to_tensor(self) -> torch.Tensor:
        state = np.zeros((2, self.board_size, self.board_size), dtype=np.float32)
        state[0] = (self.board == 1).astype(np.float32)
        state[1] = (self.board == -1).astype(np.float32)
        return torch.from_numpy(state).unsqueeze(0)

    def action_size(self) -> int:
        return self.board_size * self.board_size

    def __str__(self):
        """Pretty print the board."""
        symbols = {0: '.', 1: 'X', -1: 'O'}
        lines = []
        for i, row in enumerate(self.board):
            indent = ' ' * i
            line = indent + ' '.join(symbols[cell] for cell in row)
            lines.append(line)
        return '\n'.join(lines)


if __name__ == '__main__':
    # Test games
    print("Testing TicTacToe...")
    game = TicTacToe()
    print(f"Legal actions: {game.legal_actions()}")
    print(f"Action size: {game.action_size()}")
    print(f"Tensor shape: {game.to_tensor().shape}")

    print("\nTesting ConnectFour...")
    game = ConnectFour()
    print(f"Legal actions: {game.legal_actions()}")
    print(f"Tensor shape: {game.to_tensor().shape}")

    print("\nTesting SimpleHex...")
    game = SimpleHex(board_size=5)
    print(f"Legal actions (first 10): {game.legal_actions()[:10]}")
    print(f"Tensor shape: {game.to_tensor().shape}")
    print(f"Board:\n{game}")

    print("\n✅ All games tested successfully!")
