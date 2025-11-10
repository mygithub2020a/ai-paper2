"""
Simple game environments for testing (Tic-Tac-Toe, Connect Four).
"""

import numpy as np
from typing import Tuple, List
from .base import GameEnvironment


class TicTacToe(GameEnvironment):
    """Tic-Tac-Toe environment."""

    def __init__(self):
        self.board_size = 3
        self.reset()

    def reset(self) -> np.ndarray:
        """Reset the game."""
        self.board = np.zeros((self.board_size, self.board_size), dtype=int)
        self._current_player = 0
        self.done = False
        self.winner = None
        return self.get_state()

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, dict]:
        """Take an action."""
        if self.done:
            return self.get_state(), 0.0, True, {}

        row = action // self.board_size
        col = action % self.board_size

        if self.board[row, col] != 0:
            # Illegal move
            return self.get_state(), -10.0, True, {'illegal_move': True}

        # Make move
        self.board[row, col] = self._current_player + 1

        # Check win
        reward = 0.0
        if self._check_win():
            self.done = True
            self.winner = self._current_player
            reward = 1.0
        elif self._check_draw():
            self.done = True
            self.winner = None
            reward = 0.0
        else:
            # Switch player
            self._current_player = 1 - self._current_player

        return self.get_state(), reward, self.done, {}

    def get_legal_actions(self) -> List[int]:
        """Get legal actions."""
        legal = []
        for i in range(self.board_size):
            for j in range(self.board_size):
                if self.board[i, j] == 0:
                    legal.append(i * self.board_size + j)
        return legal

    def get_state(self) -> np.ndarray:
        """Get state representation."""
        # One-hot encoding: [player_1_positions, player_2_positions, current_player]
        state = np.zeros((3, self.board_size, self.board_size))
        state[0] = (self.board == 1).astype(float)
        state[1] = (self.board == 2).astype(float)
        state[2] = self._current_player
        return state.flatten()

    def render(self) -> str:
        """Render the board."""
        symbols = ['.', 'X', 'O']
        lines = []
        for row in self.board:
            lines.append(' '.join(symbols[x] for x in row))
        return '\n'.join(lines)

    @property
    def action_space_size(self) -> int:
        return self.board_size * self.board_size

    @property
    def observation_space_size(self) -> int:
        return 3 * self.board_size * self.board_size

    @property
    def current_player(self) -> int:
        return self._current_player

    def _check_win(self) -> bool:
        """Check if current player won."""
        player_mark = self._current_player + 1

        # Check rows
        for i in range(self.board_size):
            if np.all(self.board[i, :] == player_mark):
                return True

        # Check columns
        for j in range(self.board_size):
            if np.all(self.board[:, j] == player_mark):
                return True

        # Check diagonals
        if np.all(np.diag(self.board) == player_mark):
            return True
        if np.all(np.diag(np.fliplr(self.board)) == player_mark):
            return True

        return False

    def _check_draw(self) -> bool:
        """Check if game is a draw."""
        return np.all(self.board != 0)


class ConnectFour(GameEnvironment):
    """Connect Four environment."""

    def __init__(self, rows: int = 6, cols: int = 7):
        self.rows = rows
        self.cols = cols
        self.reset()

    def reset(self) -> np.ndarray:
        """Reset the game."""
        self.board = np.zeros((self.rows, self.cols), dtype=int)
        self._current_player = 0
        self.done = False
        self.winner = None
        return self.get_state()

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, dict]:
        """Take an action (column to drop piece)."""
        if self.done:
            return self.get_state(), 0.0, True, {}

        col = action
        if col < 0 or col >= self.cols:
            return self.get_state(), -10.0, True, {'illegal_move': True}

        # Find lowest empty row in column
        row = -1
        for r in range(self.rows - 1, -1, -1):
            if self.board[r, col] == 0:
                row = r
                break

        if row == -1:
            # Column is full (illegal move)
            return self.get_state(), -10.0, True, {'illegal_move': True}

        # Make move
        self.board[row, col] = self._current_player + 1

        # Check win
        reward = 0.0
        if self._check_win(row, col):
            self.done = True
            self.winner = self._current_player
            reward = 1.0
        elif self._check_draw():
            self.done = True
            self.winner = None
            reward = 0.0
        else:
            # Switch player
            self._current_player = 1 - self._current_player

        return self.get_state(), reward, self.done, {}

    def get_legal_actions(self) -> List[int]:
        """Get legal actions (columns that aren't full)."""
        legal = []
        for col in range(self.cols):
            if self.board[0, col] == 0:
                legal.append(col)
        return legal

    def get_state(self) -> np.ndarray:
        """Get state representation."""
        state = np.zeros((3, self.rows, self.cols))
        state[0] = (self.board == 1).astype(float)
        state[1] = (self.board == 2).astype(float)
        state[2] = self._current_player
        return state.flatten()

    def render(self) -> str:
        """Render the board."""
        symbols = ['.', 'X', 'O']
        lines = []
        for row in self.board:
            lines.append(' '.join(symbols[x] for x in row))
        lines.append(' '.join(str(i) for i in range(self.cols)))
        return '\n'.join(lines)

    @property
    def action_space_size(self) -> int:
        return self.cols

    @property
    def observation_space_size(self) -> int:
        return 3 * self.rows * self.cols

    @property
    def current_player(self) -> int:
        return self._current_player

    def _check_win(self, row: int, col: int) -> bool:
        """Check if current player won (optimized check around last move)."""
        player_mark = self._current_player + 1

        # Check horizontal
        count = 1
        # Check left
        c = col - 1
        while c >= 0 and self.board[row, c] == player_mark:
            count += 1
            c -= 1
        # Check right
        c = col + 1
        while c < self.cols and self.board[row, c] == player_mark:
            count += 1
            c += 1
        if count >= 4:
            return True

        # Check vertical
        count = 1
        # Check down
        r = row + 1
        while r < self.rows and self.board[r, col] == player_mark:
            count += 1
            r += 1
        if count >= 4:
            return True

        # Check diagonal /
        count = 1
        # Check down-left
        r, c = row + 1, col - 1
        while r < self.rows and c >= 0 and self.board[r, c] == player_mark:
            count += 1
            r += 1
            c -= 1
        # Check up-right
        r, c = row - 1, col + 1
        while r >= 0 and c < self.cols and self.board[r, c] == player_mark:
            count += 1
            r -= 1
            c += 1
        if count >= 4:
            return True

        # Check diagonal \
        count = 1
        # Check down-right
        r, c = row + 1, col + 1
        while r < self.rows and c < self.cols and self.board[r, c] == player_mark:
            count += 1
            r += 1
            c += 1
        # Check up-left
        r, c = row - 1, col - 1
        while r >= 0 and c >= 0 and self.board[r, c] == player_mark:
            count += 1
            r -= 1
            c -= 1
        if count >= 4:
            return True

        return False

    def _check_draw(self) -> bool:
        """Check if game is a draw."""
        return np.all(self.board != 0)
