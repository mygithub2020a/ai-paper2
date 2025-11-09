import torch
import torch.nn as nn
import torch.nn.functional as F

class PolicyValueNet(nn.Module):
    def __init__(self, board_size, num_actions):
        super(PolicyValueNet, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)

        self.policy_head = nn.Linear(128 * board_size * board_size, num_actions)
        self.value_head = nn.Linear(128 * board_size * board_size, 1)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = x.view(x.size(0), -1)

        policy = F.softmax(self.policy_head(x), dim=-1)
        value = torch.tanh(self.value_head(x))

        return policy, value
