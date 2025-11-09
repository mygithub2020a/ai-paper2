import torch
import torch.nn as nn

class MLP(nn.Module):
    """
    A simple multi-layer perceptron (MLP) for the modular arithmetic task.
    It takes two integer inputs, embeds them, concatenates the embeddings,
    and passes them through a series of linear layers.
    """
    def __init__(self, modulus, embed_dim, hidden_dim, num_layers, output_dim):
        """
        Args:
            modulus (int): The modulus used in the task, which determines the
                size of the embedding vocabulary.
            embed_dim (int): The dimensionality of the input embeddings.
            hidden_dim (int): The dimensionality of the hidden layers.
            num_layers (int): The number of hidden layers in the MLP.
            output_dim (int): The dimensionality of the output, which should
                typically be equal to the modulus for classification.
        """
        super().__init__()

        # Embedding layers for the two input integers
        self.embed_x = nn.Embedding(modulus, embed_dim)
        self.embed_y = nn.Embedding(modulus, embed_dim)

        input_dim = 2 * embed_dim

        layers = []
        # Input layer
        layers.append(nn.Linear(input_dim, hidden_dim))
        layers.append(nn.GELU())

        # Hidden layers
        for _ in range(num_layers - 1):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            layers.append(nn.GELU())

        self.mlp_layers = nn.Sequential(*layers)

        # Output layer
        self.output_layer = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        """
        Forward pass of the MLP.

        Args:
            x (torch.Tensor): A tensor of shape (batch_size, 2) containing
                the two integer inputs.

        Returns:
            torch.Tensor: The output logits of shape (batch_size, output_dim).
        """
        # x is expected to be of shape [batch_size, 2]
        x1 = self.embed_x(x[:, 0])
        x2 = self.embed_y(x[:, 1])

        # Concatenate the embeddings
        embedded_input = torch.cat([x1, x2], dim=1)

        hidden_output = self.mlp_layers(embedded_input)
        output_logits = self.output_layer(hidden_output)

        return output_logits

if __name__ == '__main__':
    # --- Example Usage ---
    MODULUS = 97
    BATCH_SIZE = 4

    model = MLP(
        modulus=MODULUS,
        embed_dim=64,
        hidden_dim=256,
        num_layers=3,
        output_dim=MODULUS
    )

    print("--- MLP Model ---")
    print(model)

    # Create a dummy input tensor
    dummy_input = torch.randint(0, MODULUS, (BATCH_SIZE, 2), dtype=torch.long)
    print(f"\nInput shape: {dummy_input.shape}")

    # Perform a forward pass
    output = model(dummy_input)
    print(f"Output shape: {output.shape}")

    # Check that the output shape is correct
    assert output.shape == (BATCH_SIZE, MODULUS)
    print("Output shape is correct.")
