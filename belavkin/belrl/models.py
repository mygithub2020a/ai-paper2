import torch
import torch.nn as nn
import torch.nn.functional as F

class PolicyValueNet(nn.Module):
    def __init__(self, board_size, num_actions, input_channels=6):
        super(PolicyValueNet, self).__init__()
        self.conv1 = nn.Conv2d(input_channels, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)

        self.fc_policy = nn.Linear(128 * board_size * board_size, num_actions)
        self.fc_value = nn.Linear(128 * board_size * board_size, 1)

    def forward(self, x):
        if len(x.shape) == 3: # (batch, height, width) for tictactoe
            x = x.unsqueeze(1) # Add channel dimension
        elif len(x.shape) == 4 and x.shape[1] != self.conv1.in_channels: # (batch, h, w, c) for chess
             x = x.permute(0, 3, 1, 2)

        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = x.reshape(x.size(0), -1)

        policy = F.log_softmax(self.fc_policy(x), dim=-1)
        value = torch.tanh(self.fc_value(x))

        return policy, value
