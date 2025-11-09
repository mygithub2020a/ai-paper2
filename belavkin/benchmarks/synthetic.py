import torch
import torch.nn as nn

class NonMarkovianQuadratic(nn.Module):
    """
    A non-Markovian quadratic optimization problem.
    The loss is L(theta) = 0.5 * ||theta - target||^2.
    The target moves based on the history of theta, making the problem non-Markovian.
    """
    def __init__(self, history_len=10):
        super().__init__()
        self.theta = nn.Parameter(torch.zeros(2))
        self.history = []
        self.history_len = history_len
        self.register_buffer('target', torch.ones(2))

    def forward(self):
        # Update history
        if len(self.history) >= self.history_len:
            self.history.pop(0)
        self.history.append(self.theta.detach().clone())

        # Update target based on history.
        # The target moves based on the mean of the history, creating a non-Markovian dependency.
        if len(self.history) > 1:
            history_mean = torch.stack(self.history).mean(dim=0)
            # A simple update rule to make the target move.
            self.target = torch.sin(history_mean)

        loss = 0.5 * torch.sum((self.theta - self.target)**2)
        return loss
