import torch
import torch.nn as nn

class MLP(nn.Module):
    """A configurable multi-layer perceptron."""
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers=4):
        super(MLP, self).__init__()
        layers = []
        # Input layer
        layers.append(nn.Linear(input_dim, hidden_dim))
        layers.append(nn.GELU())

        # Hidden layers
        for _ in range(num_layers - 2):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            layers.append(nn.GELU())

        # Output layer
        layers.append(nn.Linear(hidden_dim, output_dim))

        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        return self.layers(x)

class SimpleMLP(MLP):
    """A 3-layer MLP for backward compatibility."""
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(SimpleMLP, self).__init__(input_dim, hidden_dim, output_dim, num_layers=3)


class MixerBlock(nn.Module):
    """A single block of an MLP-Mixer."""
    def __init__(self, dim, num_tokens):
        super().__init__()
        self.token_mix = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(num_tokens, num_tokens),
            nn.GELU()
        )
        self.channel_mix = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, dim),
            nn.GELU()
        )

    def forward(self, x):
        # x shape: (batch_size, num_tokens, dim)
        x = x + self.token_mix(x.transpose(1, 2)).transpose(1, 2)
        x = x + self.channel_mix(x)
        return x

class MLPMixer(nn.Module):
    """A tiny MLP-Mixer model for synthetic tasks."""
    def __init__(self, input_dim, dim, output_dim, num_blocks=2):
        super().__init__()
        self.num_tokens = input_dim # Treat input dimensions as tokens

        # The "patch embedding" is just a linear projection
        self.embed = nn.Linear(self.num_tokens, dim)

        self.blocks = nn.ModuleList([
            MixerBlock(dim, self.num_tokens) for _ in range(num_blocks)
        ])

        self.head = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, output_dim)
        )

    def forward(self, x):
        # x shape: (batch_size, input_dim)
        # We need to add a dimension for the mixer blocks
        x = self.embed(x.unsqueeze(1)) # -> (batch_size, 1, dim) - wait this isn't right
        # The mixer expects (batch_size, num_tokens, dim).
        # For our data, input_dim is the number of tokens. So we need to project each token.
        # Let's adjust the embedding. The input is (batch_size, input_dim).
        # We'll treat input_dim as the number of "tokens" and project it to "dim".

        # Let's rethink the embedding for this synthetic task.
        # Input x is (batch_size, input_dim).
        # We can treat `input_dim` as the number of tokens, and use a simple embedding for each.
        # For simplicity, we can use the input as is, and project to the hidden dim.
        # This is a bit unusual, but let's make it work for the synthetic data.
        # Let's assume input_dim is small, e.g., 2 for modular addition.
        # We need to project from (batch_size, input_dim) to (batch_size, num_tokens, dim)
        # Let num_tokens = input_dim. We need a projection to dim.

        # A simple linear layer can project from input_dim to dim.
        # Let's reconsider the model architecture based on the data.

        # Let's define a simple embedding for our case.
        # The input x is (batch_size, input_dim). Let's say input_dim=2.
        # We can treat this as 2 tokens, each of dim 1.
        # Then we project these to the model's hidden dim.

        # For simplicity, a linear projection from input_dim to (num_tokens * dim)
        # seems overly complex.

        # Let's simplify:
        # A linear layer to project input_dim to the mixer's hidden dim.
        # But mixer needs a token dimension.

        # Let's keep it simple and effective:
        # We'll have a linear projection from input_dim to a flat feature vector,
        # then reshape it to fit the mixer.

        # Let's try another approach for embedding.
        # Input is (batch, input_dim). We want (batch, num_tokens, dim).
        # Let num_tokens be a hyperparameter.

        # For now, let's keep the architecture simple.
        # I will implement a standard MLP-Mixer and adjust the training harness to handle the input shape.
        # Let's assume the input will be reshaped before being passed to the model.

        # The following is a more standard MLPMixer implementation.
        # The training harness will need to adapt the data.

        self.patch_embed = nn.Linear(input_dim, dim) # A simple embedding

        self.mixer_blocks = nn.Sequential(
            *[MixerBlock(dim, 1) for _ in range(num_blocks)] # num_tokens = 1 for our simple case
        )

        self.head = nn.Linear(dim, output_dim)

    def forward(self, x):
        # Project input to the mixer dimension
        x = self.patch_embed(x)
        # Add a token dimension
        x = x.unsqueeze(1) # -> (batch_size, 1, dim)

        # Pass through mixer blocks
        x = self.mixer_blocks(x)

        # Global average pooling (over the token dim)
        x = x.mean(dim=1)

        # Final classification head
        return self.head(x)
