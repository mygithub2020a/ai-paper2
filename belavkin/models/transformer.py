import torch
import torch.nn as nn

class PositionalEncoding(nn.Module):
    """
    A simple sinusoidal positional encoding layer.
    """
    def __init__(self, d_model, max_len=50):
        super().__init__()
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-torch.log(torch.tensor(10000.0)) / d_model))
        pe = torch.zeros(max_len, d_model)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe.unsqueeze(0))

    def forward(self, x):
        return x + self.pe[:, :x.size(1)]

class TransformerModel(nn.Module):
    """
    A simple Transformer model for the modular composition task.
    """
    def __init__(self, modulus, d_model, nhead, num_encoder_layers, dim_feedforward, output_dim):
        super().__init__()

        self.embed_input = nn.Embedding(modulus, d_model)
        self.embed_func = nn.Embedding(modulus, d_model)

        self.pos_encoder = PositionalEncoding(d_model)

        encoder_layer = nn.TransformerEncoderLayer(d_model, nhead, dim_feedforward, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_encoder_layers)

        self.output_layer = nn.Linear(d_model, output_dim)

    def forward(self, x, functions):
        """
        Args:
            x (torch.Tensor): The initial input value, shape (batch_size, 1).
            functions (torch.Tensor): The sequence of functions (a, b),
                shape (batch_size, num_functions, 2).
        """
        # Embed the initial value and the function parameters
        x_embed = self.embed_input(x) # (batch_size, 1, d_model)

        # Reshape functions to (batch_size, num_functions * 2) and embed
        funcs_flat = functions.view(functions.size(0), -1)
        funcs_embed = self.embed_func(funcs_flat) # (batch_size, num_functions * 2, d_model)

        # Concatenate to form the full sequence for the transformer
        # Sequence: [x, a1, b1, a2, b2, ...]
        full_sequence = torch.cat([x_embed, funcs_embed], dim=1)

        # Add positional encoding
        seq_with_pos = self.pos_encoder(full_sequence)

        # Pass through the transformer encoder
        transformer_output = self.transformer_encoder(seq_with_pos)

        # We take the output corresponding to the first token (the initial value x)
        # as the representation of the entire sequence computation.
        final_representation = transformer_output[:, 0, :]

        # Project to the output dimension
        output_logits = self.output_layer(final_representation)

        return output_logits

if __name__ == '__main__':
    # --- Example Usage ---
    MODULUS = 97
    BATCH_SIZE = 4
    NUM_FUNCTIONS = 4

    model = TransformerModel(
        modulus=MODULUS,
        d_model=128,
        nhead=4,
        num_encoder_layers=2,
        dim_feedforward=512,
        output_dim=MODULUS
    )

    print("--- Transformer Model ---")
    print(model)

    # Create dummy input tensors
    dummy_x = torch.randint(0, MODULUS, (BATCH_SIZE, 1), dtype=torch.long)
    dummy_funcs = torch.randint(0, MODULUS, (BATCH_SIZE, NUM_FUNCTIONS, 2), dtype=torch.long)

    print(f"\nInput x shape: {dummy_x.shape}")
    print(f"Input functions shape: {dummy_funcs.shape}")

    # Perform a forward pass
    output = model(dummy_x, dummy_funcs)
    print(f"Output shape: {output.shape}")

    # Check that the output shape is correct
    assert output.shape == (BATCH_SIZE, MODULUS)
    print("Output shape is correct.")
