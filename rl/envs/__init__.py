"""
Game environments for RL benchmarking.

Provides wrappers for chess, hex, hanabi, and other board games.
"""

from .base import GameEnvironment
from .simple_games import TicTacToe, ConnectFour


def make_env(env_name: str, **kwargs):
    """
    Create a game environment.

    Args:
        env_name: Name of the environment
            - 'tictactoe': Tic-tac-toe
            - 'connect4': Connect Four
            - 'chess': Chess (requires python-chess)
            - 'hex': Hex game
            - 'hanabi': Hanabi (requires hanabi-learning-environment)

    Returns:
        Game environment instance
    """
    env_name = env_name.lower()

    if env_name == 'tictactoe':
        return TicTacToe()
    elif env_name == 'connect4':
        return ConnectFour()
    elif env_name == 'chess':
        from .chess_env import ChessEnvironment
        return ChessEnvironment(**kwargs)
    elif env_name == 'hex':
        from .hex_env import HexEnvironment
        return HexEnvironment(**kwargs)
    elif env_name == 'hanabi':
        from .hanabi_env import HanabiEnvironment
        return HanabiEnvironment(**kwargs)
    else:
        raise ValueError(f"Unknown environment: {env_name}")


__all__ = ['GameEnvironment', 'TicTacToe', 'ConnectFour', 'make_env']
