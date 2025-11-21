import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import math

class ModularAdditionDataset(Dataset):
    def __init__(self, p=113, split='train', train_fraction=0.5, seed=42):
        self.p = p
        self.data = []
        self.labels = []

        # Generate all pairs
        pairs = []
        for i in range(p):
            for j in range(p):
                pairs.append((i, j))

        # Shuffle
        g = torch.Generator()
        g.manual_seed(seed)
        indices = torch.randperm(len(pairs), generator=g).tolist()

        split_idx = int(len(pairs) * train_fraction)

        if split == 'train':
            selected_indices = indices[:split_idx]
        else:
            selected_indices = indices[split_idx:]

        for idx in selected_indices:
            a, b = pairs[idx]
            res = (a + b) % p
            self.data.append((a, b))
            self.labels.append(res)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return torch.tensor(self.data[idx]), torch.tensor(self.labels[idx])

class TransformerGrokkingModel(nn.Module):
    def __init__(self, p=113, d_model=128, nhead=4, num_layers=1, dropout=0.1):
        super().__init__()
        self.embedding = nn.Embedding(p, d_model)
        self.pos_encoder = PositionalEncoding(d_model, dropout)
        encoder_layers = nn.TransformerEncoderLayer(d_model, nhead, dim_feedforward=d_model*4, dropout=dropout, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, num_layers)
        self.decoder = nn.Linear(d_model, p)
        self.p = p

    def forward(self, x):
        # x: [batch_size, 2]
        # We treat the input as a sequence of length 2: [a, b]
        x = self.embedding(x) # [batch_size, 2, d_model]
        x = self.pos_encoder(x)
        x = self.transformer_encoder(x) # [batch_size, 2, d_model]
        # We can average the output or take the last token. Let's take the mean.
        x = x.mean(dim=1) # [batch_size, d_model]
        x = self.decoder(x) # [batch_size, p]
        return x

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=10):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x):
        """
        Args:
            x: Tensor, shape [batch_size, seq_len, embedding_dim]
        """
        x = x + self.pe[:x.size(1)].transpose(0, 1)
        return self.dropout(x)
