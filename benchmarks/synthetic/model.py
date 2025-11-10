import torch
import torch.nn as nn

class MLP(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, embedding_dim=32):
        super(MLP, self).__init__()
        self.embedding = nn.Embedding(input_size, embedding_dim)
        self.fc1 = nn.Linear(embedding_dim, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = self.embedding(x)
        x = x.squeeze(1) # Remove the extra dimension
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x
