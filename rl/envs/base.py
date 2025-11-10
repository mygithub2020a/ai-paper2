"""
Base class for game environments.
"""

from abc import ABC, abstractmethod
import numpy as np
from typing import Tuple, List, Optional


class GameEnvironment(ABC):
    """
    Abstract base class for game environments.

    All game environments should inherit from this class and implement
    the required methods.
    """

    @abstractmethod
    def reset(self) -> np.ndarray:
        """
        Reset the environment to initial state.

        Returns:
            Initial state observation
        """
        pass

    @abstractmethod
    def step(self, action: int) -> Tuple[np.ndarray, float, bool, dict]:
        """
        Take an action in the environment.

        Args:
            action: Action to take

        Returns:
            observation: Next state observation
            reward: Reward received
            done: Whether episode is done
            info: Additional information
        """
        pass

    @abstractmethod
    def get_legal_actions(self) -> List[int]:
        """
        Get list of legal actions in current state.

        Returns:
            List of legal action indices
        """
        pass

    @abstractmethod
    def get_state(self) -> np.ndarray:
        """
        Get current state representation.

        Returns:
            State as numpy array
        """
        pass

    @abstractmethod
    def render(self) -> str:
        """
        Render the current state as a string.

        Returns:
            String representation of state
        """
        pass

    @property
    @abstractmethod
    def action_space_size(self) -> int:
        """Number of possible actions."""
        pass

    @property
    @abstractmethod
    def observation_space_size(self) -> int:
        """Size of observation space."""
        pass

    @property
    @abstractmethod
    def current_player(self) -> int:
        """Current player (0 or 1 for two-player games)."""
        pass
