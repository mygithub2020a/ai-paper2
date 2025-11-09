import torch
import torch.nn as nn

class ModularArithmeticNet(nn.Module):
    """
    Network architecture for modular arithmetic tasks.
    Embedding -> Transformer -> Output
    """
    def __init__(self, p, d_model=128, n_layers=2, n_head=8):
        super(ModularArithmeticNet, self).__init__()
        self.p = p
        self.d_model = d_model

        # Embedding layer for the input numbers
        self.embedding = nn.Embedding(p, d_model)

        # Positional encoding for the sequence
        self.pos_encoder = nn.Parameter(torch.zeros(1, 3, d_model)) # Max sequence length of 3 (e.g., x, op, y)

        # Transformer Encoder
        encoder_layers = nn.TransformerEncoderLayer(d_model=d_model, nhead=n_head, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, num_layers=n_layers)

        # Output layer to predict the result
        self.fc_out = nn.Linear(d_model, p)

    def forward(self, x):
        # x is expected to be of shape (batch_size, seq_len), e.g., (batch, 2) for x, y
        # We'll treat the inputs as a sequence

        # Ensure input is long
        x = x.long()

        # Embedding
        x = self.embedding(x) * torch.sqrt(torch.tensor(self.d_model, dtype=torch.float32))

        # Add positional encoding
        x += self.pos_encoder[:, :x.size(1), :]

        # Transformer
        output = self.transformer_encoder(x)

        # We'll take the representation of the first token for the final prediction
        output = output[:, 0, :]

        # Output layer
        output = self.fc_out(output)

        return output
