import torch
import torch.nn as nn

class MLP(nn.Module):
    """
    A simple Multi-Layer Perceptron (MLP) with an embedding layer for tabular data.
    """
    def __init__(self, vocab_size, embedding_dim, hidden_dim, output_dim, num_layers=4, activation=nn.GELU):
        """
        Initializes the MLP model.

        Args:
            vocab_size (int): The size of the vocabulary for the embedding layer (typically the modulus).
            embedding_dim (int): The dimension of the embeddings.
            hidden_dim (int): The dimension of the hidden layers.
            output_dim (int): The dimension of the output layer (typically the modulus).
            num_layers (int, optional): The number of hidden layers. Defaults to 4.
            activation (nn.Module, optional): The activation function to use. Defaults to nn.GELU.
        """
        super().__init__()

        self.embedding = nn.Embedding(vocab_size, embedding_dim)

        layers = []
        # The input to the first linear layer is the concatenated embeddings of the 4 input integers
        input_size = 4 * embedding_dim

        # First hidden layer
        layers.append(nn.Linear(input_size, hidden_dim))
        layers.append(activation())

        # Additional hidden layers
        for _ in range(num_layers - 1):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            layers.append(activation())

        # Output layer
        layers.append(nn.Linear(hidden_dim, output_dim))

        self.mlp = nn.Sequential(*layers)

    def forward(self, x):
        """
        Performs the forward pass of the model.

        Args:
            x (torch.Tensor): The input tensor of shape (batch_size, 4).

        Returns:
            torch.Tensor: The output tensor of shape (batch_size, output_dim).
        """
        # x shape: (batch_size, 4)
        embedded = self.embedding(x)  # -> (batch_size, 4, embedding_dim)

        # Flatten the embeddings to be fed into the MLP
        flattened = embedded.view(embedded.size(0), -1)  # -> (batch_size, 4 * embedding_dim)

        return self.mlp(flattened)

if __name__ == '__main__':
    # Example usage:
    modulus = 97
    embedding_dim = 32
    hidden_dim = 256
    num_layers = 4
    batch_size = 64

    model = MLP(
        vocab_size=modulus,
        embedding_dim=embedding_dim,
        hidden_dim=hidden_dim,
        output_dim=modulus,
        num_layers=num_layers
    )

    print("MLP Model Architecture:")
    print(model)

    # Create a dummy input tensor
    dummy_input = torch.randint(0, modulus, (batch_size, 4), dtype=torch.long)
    print(f"\nDummy input shape: {dummy_input.shape}")

    # Perform a forward pass
    output = model(dummy_input)
    print(f"Output shape: {output.shape}")

    # Check that the output shape is correct
    assert output.shape == (batch_size, modulus)
    print("\nModel created and tested successfully.")
