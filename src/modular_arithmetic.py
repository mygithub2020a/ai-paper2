import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

class ModularArithmeticDataset(Dataset):
    def __init__(self, p=113, train_split=0.8, seed=42, mode='train'):
        self.p = p
        self.mode = mode

        # Generate all pairs
        pairs = []
        for i in range(p):
            for j in range(p):
                pairs.append((i, j, (i + j) % p))

        # Shuffle and split
        generator = torch.Generator().manual_seed(seed)
        indices = torch.randperm(len(pairs), generator=generator).tolist()

        split_idx = int(len(pairs) * train_split)
        if mode == 'train':
            self.data = [pairs[i] for i in indices[:split_idx]]
        else:
            self.data = [pairs[i] for i in indices[split_idx:]]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        a, b, res = self.data[idx]
        return torch.tensor([a, b]), torch.tensor(res)

class TransformerModel(nn.Module):
    def __init__(self, vocab_size, embed_dim=128, num_heads=4, num_layers=1):
        super(TransformerModel, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.pos_embedding = nn.Parameter(torch.randn(1, 2, embed_dim))

        encoder_layer = nn.TransformerEncoderLayer(d_model=embed_dim, nhead=num_heads, batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        self.decoder = nn.Linear(embed_dim, vocab_size)

    def forward(self, x):
        # x: (batch_size, 2)
        x = self.embedding(x) + self.pos_embedding
        x = self.transformer(x)
        # Mean pooling
        x = x.mean(dim=1)
        x = self.decoder(x)
        return x
