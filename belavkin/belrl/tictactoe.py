import numpy as np

class TicTacToeGame:
    def __init__(self):
        self.board = np.zeros((3, 3), dtype=int)
        self.action_size = 9

    def get_board_state(self):
        return self.board

    def get_legal_moves(self, board):
        return list(np.argwhere(board.flatten() == 0).flatten())

    def make_move(self, action):
        row, col = action // 3, action % 3
        player = 1 if self.board.sum() == 0 else -1
        if self.board[row, col] == 0:
            self.board[row, col] = player

    def is_game_over(self, board):
        # Check rows, cols, and diagonals for a win
        for i in range(3):
            if abs(board[i, :].sum()) == 3 or abs(board[:, i].sum()) == 3:
                return True
        if abs(np.diag(board).sum()) == 3 or abs(np.diag(np.fliplr(board)).sum()) == 3:
            return True
        # Check for a draw
        if len(self.get_legal_moves(board)) == 0:
            return True
        return False

    def get_game_result(self, board):
        for i in range(3):
            if board[i, :].sum() == 3 or board[:, i].sum() == 3:
                return 1
            if board[i, :].sum() == -3 or board[:, i].sum() == -3:
                return -1
        if np.diag(board).sum() == 3 or np.diag(np.fliplr(board)).sum() == 3:
            return 1
        if np.diag(board).sum() == -3 or np.diag(np.fliplr(board)).sum() == -3:
            return -1
        return 0

    def get_next_state(self, board, player, action):
        next_board = board.copy()
        row, col = action // 3, action % 3
        next_board[row, col] = player
        return next_board, -player

    def get_canonical_form(self, board, player):
        return board * player
