"""
Utility functions for benchmarking.
"""

import torch
import numpy as np
import random
import json
import pickle
from pathlib import Path
from typing import Dict, Any, Optional


def set_seed(seed: int):
    """Set random seeds for reproducibility."""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def save_results(results: Dict[str, Any], filepath: str):
    """
    Save benchmark results to file.

    Args:
        results: Dictionary containing benchmark results
        filepath: Path to save file (.json or .pkl)
    """
    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)

    if filepath.suffix == '.json':
        # Convert numpy arrays to lists for JSON serialization
        json_results = {}
        for key, value in results.items():
            if isinstance(value, np.ndarray):
                json_results[key] = value.tolist()
            elif isinstance(value, torch.Tensor):
                json_results[key] = value.cpu().numpy().tolist()
            else:
                json_results[key] = value

        with open(filepath, 'w') as f:
            json.dump(json_results, f, indent=2)
    else:
        with open(filepath, 'wb') as f:
            pickle.dump(results, f)


def load_results(filepath: str) -> Dict[str, Any]:
    """
    Load benchmark results from file.

    Args:
        filepath: Path to results file

    Returns:
        Dictionary containing results
    """
    filepath = Path(filepath)

    if filepath.suffix == '.json':
        with open(filepath, 'r') as f:
            return json.load(f)
    else:
        with open(filepath, 'rb') as f:
            return pickle.load(f)


class MetricsTracker:
    """Track and compute metrics during training."""

    def __init__(self):
        self.reset()

    def reset(self):
        """Reset all tracked metrics."""
        self.losses = []
        self.accuracies = []
        self.grad_norms = []

    def update(
        self,
        loss: float,
        accuracy: Optional[float] = None,
        grad_norm: Optional[float] = None,
    ):
        """Update tracked metrics."""
        self.losses.append(loss)
        if accuracy is not None:
            self.accuracies.append(accuracy)
        if grad_norm is not None:
            self.grad_norms.append(grad_norm)

    def get_average_loss(self) -> float:
        """Get average loss."""
        return np.mean(self.losses) if self.losses else 0.0

    def get_average_accuracy(self) -> float:
        """Get average accuracy."""
        return np.mean(self.accuracies) if self.accuracies else 0.0

    def get_average_grad_norm(self) -> float:
        """Get average gradient norm."""
        return np.mean(self.grad_norms) if self.grad_norms else 0.0

    def get_metrics(self) -> Dict[str, float]:
        """Get all metrics as a dictionary."""
        return {
            'loss': self.get_average_loss(),
            'accuracy': self.get_average_accuracy(),
            'grad_norm': self.get_average_grad_norm(),
        }


def compute_accuracy(predictions: torch.Tensor, targets: torch.Tensor) -> float:
    """
    Compute classification accuracy.

    Args:
        predictions: Model predictions (logits)
        targets: Ground truth labels

    Returns:
        Accuracy as a float
    """
    pred_labels = predictions.argmax(dim=1)
    correct = (pred_labels == targets).float().sum()
    accuracy = correct / len(targets)
    return accuracy.item()


def compute_gradient_norm(model: torch.nn.Module) -> float:
    """
    Compute the L2 norm of gradients.

    Args:
        model: PyTorch model

    Returns:
        Gradient norm as a float
    """
    total_norm = 0.0
    for p in model.parameters():
        if p.grad is not None:
            param_norm = p.grad.data.norm(2)
            total_norm += param_norm.item() ** 2
    total_norm = total_norm ** 0.5
    return total_norm


def get_device() -> torch.device:
    """Get the best available device (CUDA if available, else CPU)."""
    return torch.device('cuda' if torch.cuda.is_available() else 'cpu')
