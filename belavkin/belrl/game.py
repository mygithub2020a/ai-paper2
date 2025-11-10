import chess
import numpy as np

class ChessGame:
    def __init__(self):
        self.board = chess.Board()
        self.move_map, self.inv_move_map = self._create_move_maps()
        self.action_size = len(self.move_map)

    def _create_move_maps(self):
        """Creates mappings from all possible moves to an integer index and back."""
        move_map = {}
        idx = 0
        for from_square in range(64):
            for to_square in range(64):
                # Standard moves
                move = chess.Move(from_square, to_square)
                if move not in move_map:
                    move_map[move] = idx
                    idx += 1

                # Promotion moves
                for promotion_piece in [chess.KNIGHT, chess.BISHOP, chess.ROOK, chess.QUEEN]:
                    promo_move = chess.Move(from_square, to_square, promotion=promotion_piece)
                    if promo_move not in move_map:
                        move_map[promo_move] = idx
                        idx += 1

        inv_move_map = {v: k for k, v in move_map.items()}
        return move_map, inv_move_map

    def get_board_state(self):
        """Returns the current board state as a numpy array."""
        return self.get_board_state_from_board(self.board)

    def get_legal_moves(self, board):
        """Returns a list of legal moves as integer indices."""
        legal_moves = []
        for move in board.legal_moves:
            if move in self.move_map:
                legal_moves.append(self.move_map[move])
        return legal_moves

    def make_move(self, action):
        """Makes a move on the board given an integer action."""
        move = self.inv_move_map[action]
        self.board.push(move)

    def is_game_over(self, board):
        """Checks if the game is over."""
        return board.is_game_over()

    def get_game_result(self, board):
        """Gets the result of the game."""
        if board.is_checkmate():
            return 1 if board.turn == chess.BLACK else -1
        return 0  # Draw

    def get_next_state(self, board, player, action):
        """
        Returns the next state of the board after applying the action.
        This is a helper function for MCTS.
        """
        next_board = board.copy()
        move = self.inv_move_map[action]
        next_board.push(move)
        return next_board, -player

    def get_canonical_form(self, board, player):
        """
        Returns the canonical form of the board state.
        The canonical form is from the perspective of the current player.
        """
        return self.get_board_state_from_board(board) * (1 if player == chess.WHITE else -1)

    def get_board_state_from_board(self, board):
        """Helper function to get the state from a chess.Board object."""
        state = np.zeros((8, 8, 6), dtype=np.float32)
        for i in range(64):
            piece = board.piece_at(i)
            if piece:
                color = 1 if piece.color == chess.WHITE else -1
                piece_type = piece.piece_type - 1
                state[i // 8, i % 8, piece_type] = color
        return state
