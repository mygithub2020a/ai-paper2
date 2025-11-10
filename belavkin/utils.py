"""Utility functions for training and evaluation."""

import torch
import numpy as np
import time
from typing import Optional, Dict, Any
import json
import os


def set_seed(seed: int):
    """Set random seeds for reproducibility."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def count_parameters(model: torch.nn.Module) -> int:
    """Count total trainable parameters."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def get_device() -> torch.device:
    """Get the best available device."""
    if torch.cuda.is_available():
        return torch.device('cuda')
    elif torch.backends.mps.is_available():
        return torch.device('mps')
    else:
        return torch.device('cpu')


class AverageMeter:
    """Computes and stores the average and current value."""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


class Timer:
    """Simple timer for measuring execution time."""

    def __init__(self):
        self.times = {}
        self.start_times = {}

    def start(self, name: str):
        self.start_times[name] = time.time()

    def stop(self, name: str) -> float:
        if name not in self.start_times:
            raise ValueError(f"Timer '{name}' was never started")
        elapsed = time.time() - self.start_times[name]
        if name not in self.times:
            self.times[name] = []
        self.times[name].append(elapsed)
        return elapsed

    def get_average(self, name: str) -> float:
        if name not in self.times:
            return 0.0
        return np.mean(self.times[name])


def save_checkpoint(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    metrics: Dict[str, float],
    path: str,
):
    """Save model checkpoint."""
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'metrics': metrics,
    }
    torch.save(checkpoint, path)


def load_checkpoint(
    model: torch.nn.Module,
    optimizer: Optional[torch.optim.Optimizer],
    path: str,
) -> Dict[str, Any]:
    """Load model checkpoint."""
    checkpoint = torch.load(path)
    model.load_state_dict(checkpoint['model_state_dict'])
    if optimizer is not None:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    return checkpoint


def calculate_accuracy(predictions: torch.Tensor, labels: torch.Tensor, modulus: int) -> float:
    """
    Calculate accuracy for modular arithmetic tasks.

    Args:
        predictions: Model predictions (logits or values)
        labels: Ground truth labels
        modulus: Modulus for classification

    Returns:
        Accuracy as a percentage
    """
    if predictions.dim() > 1 and predictions.shape[1] > 1:
        # Classification: take argmax
        pred_classes = predictions.argmax(dim=1)
    else:
        # Regression: round to nearest integer mod modulus
        pred_classes = torch.round(predictions.squeeze()) % modulus

    correct = (pred_classes == labels).float().sum()
    accuracy = 100.0 * correct / labels.numel()
    return accuracy.item()


class Logger:
    """Simple logger for training metrics."""

    def __init__(self, log_dir: str, experiment_name: str):
        self.log_dir = log_dir
        self.experiment_name = experiment_name
        self.log_file = os.path.join(log_dir, f"{experiment_name}.json")

        os.makedirs(log_dir, exist_ok=True)

        self.logs = []

    def log(self, metrics: Dict[str, Any], step: Optional[int] = None):
        """Log metrics."""
        entry = {
            'timestamp': time.time(),
            'step': step if step is not None else len(self.logs),
            **metrics
        }
        self.logs.append(entry)

        # Save to file
        with open(self.log_file, 'w') as f:
            json.dump(self.logs, f, indent=2)

    def get_logs(self):
        """Get all logged metrics."""
        return self.logs


def format_time(seconds: float) -> str:
    """Format seconds as human-readable time."""
    if seconds < 60:
        return f"{seconds:.2f}s"
    elif seconds < 3600:
        minutes = seconds / 60
        return f"{minutes:.2f}m"
    else:
        hours = seconds / 3600
        return f"{hours:.2f}h"


def print_metrics(
    epoch: int,
    train_loss: float,
    train_acc: float,
    test_loss: float,
    test_acc: float,
    epoch_time: float,
):
    """Print training metrics in a formatted way."""
    print(f"Epoch {epoch:3d} | "
          f"Train Loss: {train_loss:.4f} Acc: {train_acc:6.2f}% | "
          f"Test Loss: {test_loss:.4f} Acc: {test_acc:6.2f}% | "
          f"Time: {format_time(epoch_time)}")


if __name__ == '__main__':
    # Test utilities
    set_seed(42)
    device = get_device()
    print(f"Device: {device}")

    # Test timer
    timer = Timer()
    timer.start('test')
    time.sleep(0.1)
    elapsed = timer.stop('test')
    print(f"Elapsed: {format_time(elapsed)}")

    # Test logger
    logger = Logger('./test_logs', 'test_exp')
    logger.log({'loss': 0.5, 'acc': 95.0}, step=0)
    logger.log({'loss': 0.3, 'acc': 97.0}, step=1)
    print(f"Logged {len(logger.get_logs())} entries")
