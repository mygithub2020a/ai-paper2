"""
Monte Carlo Tree Search (MCTS) for BelRL.

Implements AlphaZero-style MCTS with neural network guidance.
"""

import math
import numpy as np
from dataclasses import dataclass
from typing import Optional, List, Tuple, Dict, Any
from collections import defaultdict


@dataclass
class MCTSConfig:
    """Configuration for MCTS."""

    num_simulations: int = 800
    c_puct: float = 1.0  # Exploration constant
    temperature: float = 1.0  # Temperature for action selection
    dirichlet_alpha: float = 0.3  # Dirichlet noise alpha
    dirichlet_epsilon: float = 0.25  # Dirichlet noise weight
    add_dirichlet_noise: bool = True


class MCTSNode:
    """Node in the MCTS tree."""

    def __init__(self, prior: float, to_play: int):
        """
        Args:
            prior: Prior probability from policy network
            to_play: Player to move (1 or -1)
        """
        self.visit_count = 0
        self.to_play = to_play
        self.prior = prior
        self.value_sum = 0.0
        self.children: Dict[Any, MCTSNode] = {}

    def expanded(self) -> bool:
        """Check if node has been expanded."""
        return len(self.children) > 0

    def value(self) -> float:
        """Return the value of the node."""
        if self.visit_count == 0:
            return 0.0
        return self.value_sum / self.visit_count

    def select_action(self, config: MCTSConfig) -> Any:
        """
        Select action using PUCT algorithm.

        Args:
            config: MCTS configuration

        Returns:
            Selected action
        """
        best_score = -float('inf')
        best_action = None

        # Calculate UCB scores for all children
        for action, child in self.children.items():
            score = self._ucb_score(child, config.c_puct)
            if score > best_score:
                best_score = score
                best_action = action

        return best_action

    def _ucb_score(self, child: 'MCTSNode', c_puct: float) -> float:
        """
        Calculate UCB score for a child node.

        UCB = Q(s,a) + c_puct * P(s,a) * sqrt(N(s)) / (1 + N(s,a))
        """
        prior_score = c_puct * child.prior * math.sqrt(self.visit_count) / (1 + child.visit_count)

        if child.visit_count > 0:
            # Value from perspective of current player
            value_score = -child.value()
        else:
            value_score = 0.0

        return value_score + prior_score

    def expand(self, actions: List[Any], policy_probs: np.ndarray, to_play: int):
        """
        Expand node with children.

        Args:
            actions: List of legal actions
            policy_probs: Policy probabilities for each action
            to_play: Player to move next
        """
        for action, prob in zip(actions, policy_probs):
            if action not in self.children:
                self.children[action] = MCTSNode(prior=prob, to_play=to_play)

    def add_dirichlet_noise(self, config: MCTSConfig):
        """Add Dirichlet noise to root node for exploration."""
        actions = list(self.children.keys())
        noise = np.random.dirichlet([config.dirichlet_alpha] * len(actions))

        for action, noise_value in zip(actions, noise):
            self.children[action].prior = (
                self.children[action].prior * (1 - config.dirichlet_epsilon) +
                noise_value * config.dirichlet_epsilon
            )


class MCTS:
    """Monte Carlo Tree Search."""

    def __init__(self, config: MCTSConfig):
        self.config = config

    def run_simulations(
        self,
        state: Any,
        network,
        num_simulations: Optional[int] = None,
    ) -> Tuple[Dict[Any, float], MCTSNode]:
        """
        Run MCTS simulations from the given state.

        Args:
            state: Game state
            network: Policy-value network
            num_simulations: Number of simulations (overrides config)

        Returns:
            action_probs: Dictionary mapping actions to visit counts
            root: Root node of the search tree
        """
        num_sims = num_simulations if num_simulations is not None else self.config.num_simulations

        # Create root node
        root = MCTSNode(prior=1.0, to_play=state.to_play())

        # Add Dirichlet noise to root for exploration
        if self.config.add_dirichlet_noise:
            self._add_exploration_noise(root, state, network)

        # Run simulations
        for _ in range(num_sims):
            node = root
            search_path = [node]
            current_state = state.clone()

            # Selection: traverse tree until we find unexpanded node
            while node.expanded():
                action = node.select_action(self.config)
                current_state.apply_action(action)
                node = node.children[action]
                search_path.append(node)

            # Check terminal state
            terminal, value = current_state.is_terminal()

            if not terminal:
                # Expansion: expand node with network predictions
                value = self._expand_node(node, current_state, network)

            # Backpropagation: update values along search path
            self._backpropagate(search_path, value, state.to_play())

        # Extract action probabilities from visit counts
        action_probs = self._get_action_probs(root)

        return action_probs, root

    def _add_exploration_noise(self, root: MCTSNode, state: Any, network):
        """Add Dirichlet noise to root node."""
        legal_actions = state.legal_actions()

        # Get policy from network
        policy_logits, _ = network(state.to_tensor())
        policy_probs = self._mask_invalid_actions(
            policy_logits.exp().detach().numpy()[0],
            legal_actions,
            state.action_size()
        )

        # Expand root
        root.expand(legal_actions, policy_probs, state.to_play())

        # Add noise
        root.add_dirichlet_noise(self.config)

    def _expand_node(self, node: MCTSNode, state: Any, network) -> float:
        """
        Expand node using network predictions.

        Returns:
            Value estimate from network
        """
        legal_actions = state.legal_actions()

        # Get network predictions
        policy_logits, value = network(state.to_tensor())

        # Mask illegal actions
        policy_probs = self._mask_invalid_actions(
            policy_logits.exp().detach().numpy()[0],
            legal_actions,
            state.action_size()
        )

        # Expand node
        node.expand(legal_actions, policy_probs, -state.to_play())

        # Return value from perspective of player to move
        return value.item()

    def _mask_invalid_actions(
        self,
        policy: np.ndarray,
        legal_actions: List[int],
        action_size: int
    ) -> np.ndarray:
        """Mask invalid actions and renormalize."""
        mask = np.zeros(action_size)
        mask[legal_actions] = 1.0

        masked_policy = policy * mask

        # Renormalize
        policy_sum = masked_policy.sum()
        if policy_sum > 0:
            masked_policy /= policy_sum
        else:
            # Uniform over legal actions if all policies were masked
            masked_policy[legal_actions] = 1.0 / len(legal_actions)

        return masked_policy[legal_actions]

    def _backpropagate(self, search_path: List[MCTSNode], value: float, to_play: int):
        """Backpropagate value up the search path."""
        for node in reversed(search_path):
            node.value_sum += value if node.to_play == to_play else -value
            node.visit_count += 1

    def _get_action_probs(self, root: MCTSNode) -> Dict[Any, float]:
        """
        Get action probabilities from root visit counts.

        Uses temperature-based sampling.
        """
        visit_counts = {
            action: child.visit_count
            for action, child in root.children.items()
        }

        if self.config.temperature == 0:
            # Greedy: select most visited action
            max_count = max(visit_counts.values())
            probs = {
                action: 1.0 if count == max_count else 0.0
                for action, count in visit_counts.items()
            }
        else:
            # Temperature-based sampling
            total = sum(visit_counts.values())
            if total == 0:
                # Uniform if no visits (shouldn't happen)
                probs = {action: 1.0 / len(visit_counts) for action in visit_counts}
            else:
                # Apply temperature
                temp_counts = {
                    action: count ** (1.0 / self.config.temperature)
                    for action, count in visit_counts.items()
                }
                temp_sum = sum(temp_counts.values())
                probs = {action: count / temp_sum for action, count in temp_counts.items()}

        return probs

    def select_action(
        self,
        state: Any,
        network,
        deterministic: bool = False
    ) -> Any:
        """
        Select action using MCTS.

        Args:
            state: Current game state
            network: Policy-value network
            deterministic: If True, select greedily; otherwise sample

        Returns:
            Selected action
        """
        action_probs, _ = self.run_simulations(state, network)

        if deterministic:
            # Select action with highest probability
            return max(action_probs.items(), key=lambda x: x[1])[0]
        else:
            # Sample action according to probabilities
            actions = list(action_probs.keys())
            probs = list(action_probs.values())
            return np.random.choice(actions, p=probs)


class GameState:
    """
    Abstract base class for game states.

    Subclasses should implement these methods for specific games.
    """

    def clone(self) -> 'GameState':
        """Create a copy of the state."""
        raise NotImplementedError

    def apply_action(self, action: Any):
        """Apply an action to the state (modifies in-place)."""
        raise NotImplementedError

    def legal_actions(self) -> List[Any]:
        """Return list of legal actions."""
        raise NotImplementedError

    def is_terminal(self) -> Tuple[bool, float]:
        """
        Check if state is terminal.

        Returns:
            terminal: True if game is over
            value: Game outcome from perspective of current player
        """
        raise NotImplementedError

    def to_play(self) -> int:
        """Return player to move (1 or -1)."""
        raise NotImplementedError

    def to_tensor(self):
        """Convert state to neural network input tensor."""
        raise NotImplementedError

    def action_size(self) -> int:
        """Return total number of possible actions."""
        raise NotImplementedError


if __name__ == '__main__':
    print("MCTS implementation complete!")
    print("To use MCTS, implement the GameState interface for your game.")
