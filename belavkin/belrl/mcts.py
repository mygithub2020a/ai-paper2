import math
import numpy as np
import torch

class MCTS:
    def __init__(self, game, model, args):
        self.game = game
        self.model = model
        self.args = args
        self.Qsa = {}  # Stores Q values for s,a pairs
        self.Nsa = {}  # Stores # times edge s,a was visited
        self.Ns = {}   # Stores # times board s was visited
        self.Ps = {}   # Stores initial policy (returned by neural net)

    def get_action_prob(self, board, player, temp=1):
        """
        This function performs num_simulations MCTS simulations starting from
        the given board.

        Returns:
            probs: a policy vector where the probability of the ith action is
                   proportional to Nsa[(s,a)]**(1./temp)
        """
        for _ in range(self.args.num_simulations):
            self.search(board, player)

        s = str(board.tobytes())
        counts = [self.Nsa.get((s, a), 0) for a in range(self.game.action_size)]

        if temp == 0:
            best_as = np.array(np.argwhere(counts == np.max(counts))).flatten()
            best_a = np.random.choice(best_as)
            probs = [0] * len(counts)
            probs[best_a] = 1
            return probs

        counts = [x**(1. / temp) for x in counts]
        counts_sum = float(sum(counts))
        if counts_sum == 0:
            return [1.0/len(counts)] * len(counts)
        probs = [x / counts_sum for x in counts]
        return probs

    def search(self, board, player):
        """
        This function performs one iteration of MCTS. It is recursively called
        till a leaf node is found. The action chosen at each node is one that
        has the maximum upper confidence bound (UCB).
        """
        s = str(board.tobytes())

        # If the game has ended, return the reward
        if self.game.is_game_over(board):
            return -self.game.get_game_result(board)

        if s not in self.Ps:
            # This is a leaf node; expand it
            canonical_board = self.game.get_canonical_form(board, player)
            board_tensor = torch.tensor(canonical_board, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
            self.Ps[s], v = self.model(board_tensor)
            self.Ps[s] = torch.exp(self.Ps[s]).squeeze().detach().cpu().numpy()
            self.Ns[s] = 0
            return -v.item()

        # Select the action with the highest UCB
        cur_best = -float('inf')
        best_act = -1

        for a in self.game.get_legal_moves(board):
            if (s, a) in self.Qsa:
                u = self.Qsa[(s, a)] + self.args.cpuct * self.Ps[s][a] * math.sqrt(self.Ns[s]) / (1 + self.Nsa[(s, a)])
            else:
                u = self.args.cpuct * self.Ps[s][a] * math.sqrt(self.Ns[s] + 1e-8)

            if u > cur_best:
                cur_best = u
                best_act = a

        a = best_act
        next_board, next_player = self.game.get_next_state(board, player, a)

        v = self.search(next_board, next_player)

        # Backpropagate the value
        if (s, a) in self.Qsa:
            self.Qsa[(s, a)] = (self.Nsa[(s, a)] * self.Qsa[(s, a)] + v) / (self.Nsa[(s, a)] + 1)
            self.Nsa[(s, a)] += 1
        else:
            self.Qsa[(s, a)] = v
            self.Nsa[(s, a)] = 1

        self.Ns[s] += 1
        return -v
