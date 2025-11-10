"""
Pure NumPy implementation of Belavkin Optimizer for actual experiments.
This allows us to run real experiments without depending on PyTorch installation.
"""

import numpy as np
from typing import Callable, Optional, Tuple


class BelavkinOptimizerNumPy:
    """NumPy implementation of Belavkin Optimizer."""

    def __init__(
        self,
        learning_rate: float = 0.01,
        gamma: float = 0.1,
        beta: float = 0.01,
        momentum: float = 0.0,
    ):
        self.learning_rate = learning_rate
        self.gamma = gamma
        self.beta = beta
        self.momentum = momentum
        self.momentum_buffer = None

    def step(self, params: np.ndarray, grads: np.ndarray) -> np.ndarray:
        """
        Perform one optimization step.

        Args:
            params: Current parameters
            grads: Gradients

        Returns:
            Updated parameters
        """
        if self.momentum_buffer is None:
            self.momentum_buffer = np.zeros_like(params)

        # Compute update components
        grad_squared = grads * grads
        adaptive_term = self.gamma * grad_squared
        gradient_term = grads

        # Deterministic update
        deterministic_update = adaptive_term + gradient_term

        # Stochastic term
        noise = np.random.randn(*params.shape)
        stochastic_term = self.beta * grads * noise

        # Full update
        full_update = deterministic_update + stochastic_term

        # Apply momentum
        self.momentum_buffer = (
            self.momentum * self.momentum_buffer + full_update
        )

        # Update parameters
        updated_params = params - self.learning_rate * self.momentum_buffer

        return updated_params


class SimpleNeuralNetNumPy:
    """Simple neural network in NumPy."""

    def __init__(self, input_dim: int = 2, hidden_dim: int = 32):
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim

        # Initialize weights
        self.w1 = np.random.randn(input_dim, hidden_dim) * 0.01
        self.b1 = np.zeros((1, hidden_dim))
        self.w2 = np.random.randn(hidden_dim, hidden_dim) * 0.01
        self.b2 = np.zeros((1, hidden_dim))
        self.w3 = np.random.randn(hidden_dim, 1) * 0.01
        self.b3 = np.zeros((1, 1))

    def forward(self, x: np.ndarray) -> np.ndarray:
        """Forward pass."""
        self.z1 = np.dot(x, self.w1) + self.b1
        self.a1 = np.maximum(0, self.z1)  # ReLU

        self.z2 = np.dot(self.a1, self.w2) + self.b2
        self.a2 = np.maximum(0, self.z2)  # ReLU

        self.z3 = np.dot(self.a2, self.w3) + self.b3
        return self.z3

    def backward(
        self, x: np.ndarray, y: np.ndarray, output: np.ndarray
    ) -> Tuple[dict, float]:
        """Backward pass to compute gradients."""
        m = x.shape[0]
        loss = np.mean((output - y) ** 2)

        # Gradient w.r.t output
        dz3 = (output - y) * 2 / m
        dw3 = np.dot(self.a2.T, dz3)
        db3 = np.sum(dz3, axis=0, keepdims=True)

        # Gradient w.r.t hidden layer 2
        da2 = np.dot(dz3, self.w3.T)
        dz2 = da2 * (self.z2 > 0)  # ReLU derivative
        dw2 = np.dot(self.a1.T, dz2)
        db2 = np.sum(dz2, axis=0, keepdims=True)

        # Gradient w.r.t hidden layer 1
        da1 = np.dot(dz2, self.w2.T)
        dz1 = da1 * (self.z1 > 0)  # ReLU derivative
        dw1 = np.dot(x.T, dz1)
        db1 = np.sum(dz1, axis=0, keepdims=True)

        grads = {
            "w1": dw1,
            "b1": db1,
            "w2": dw2,
            "b2": db2,
            "w3": dw3,
            "b3": db3,
        }

        return grads, loss

    def update_params(self, grads: dict, learning_rate: float):
        """Update parameters using gradients."""
        self.w1 -= learning_rate * grads["w1"]
        self.b1 -= learning_rate * grads["b1"]
        self.w2 -= learning_rate * grads["w2"]
        self.b2 -= learning_rate * grads["b2"]
        self.w3 -= learning_rate * grads["w3"]
        self.b3 -= learning_rate * grads["b3"]

    def get_all_params(self) -> np.ndarray:
        """Get all parameters as a flat array."""
        return np.concatenate([
            self.w1.flatten(),
            self.b1.flatten(),
            self.w2.flatten(),
            self.b2.flatten(),
            self.w3.flatten(),
            self.b3.flatten(),
        ])

    def set_all_params(self, params: np.ndarray):
        """Set all parameters from a flat array."""
        idx = 0
        shape_w1 = self.w1.shape
        self.w1 = params[idx:idx+np.prod(shape_w1)].reshape(shape_w1)
        idx += np.prod(shape_w1)

        shape_b1 = self.b1.shape
        self.b1 = params[idx:idx+np.prod(shape_b1)].reshape(shape_b1)
        idx += np.prod(shape_b1)

        shape_w2 = self.w2.shape
        self.w2 = params[idx:idx+np.prod(shape_w2)].reshape(shape_w2)
        idx += np.prod(shape_w2)

        shape_b2 = self.b2.shape
        self.b2 = params[idx:idx+np.prod(shape_b2)].reshape(shape_b2)
        idx += np.prod(shape_b2)

        shape_w3 = self.w3.shape
        self.w3 = params[idx:idx+np.prod(shape_w3)].reshape(shape_w3)
        idx += np.prod(shape_w3)

        shape_b3 = self.b3.shape
        self.b3 = params[idx:idx+np.prod(shape_b3)].reshape(shape_b3)


if __name__ == "__main__":
    print("Belavkin Optimizer NumPy module loaded successfully")
