"""
Training and evaluation utilities for benchmarking optimizers.
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from typing import Dict, List, Optional, Tuple
from tqdm import tqdm
import time


def train_model(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    train_loader: DataLoader,
    val_loader: DataLoader,
    num_epochs: int,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
    criterion: Optional[nn.Module] = None,
    log_interval: int = 10,
    early_stopping_patience: Optional[int] = None,
) -> Dict[str, List[float]]:
    """
    Train a model with given optimizer and track metrics.

    Args:
        model: Neural network model
        optimizer: Optimizer instance
        train_loader: Training data loader
        val_loader: Validation data loader
        num_epochs: Number of training epochs
        device: Device to train on
        criterion: Loss function (default: CrossEntropyLoss)
        log_interval: Logging frequency in epochs
        early_stopping_patience: Stop if no improvement for N epochs (None to disable)

    Returns:
        Dictionary containing training history:
            - train_loss: Training loss per epoch
            - train_acc: Training accuracy per epoch
            - val_loss: Validation loss per epoch
            - val_acc: Validation accuracy per epoch
            - epoch_time: Time per epoch
            - best_epoch: Epoch with best validation accuracy
            - convergence_epoch: Epoch reaching 95% val accuracy (if any)
    """
    model.to(device)

    if criterion is None:
        criterion = nn.CrossEntropyLoss()

    history = {
        "train_loss": [],
        "train_acc": [],
        "val_loss": [],
        "val_acc": [],
        "epoch_time": [],
        "best_epoch": 0,
        "best_val_acc": 0.0,
        "convergence_epoch": None,  # First epoch reaching 95% accuracy
    }

    best_val_acc = 0.0
    patience_counter = 0

    for epoch in range(num_epochs):
        epoch_start = time.time()

        # Training phase
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0

        for batch_idx, (inputs, targets) in enumerate(train_loader):
            inputs, targets = inputs.to(device), targets.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * inputs.size(0)
            _, predicted = outputs.max(1)
            train_total += targets.size(0)
            train_correct += predicted.eq(targets).sum().item()

        train_loss = train_loss / train_total
        train_acc = 100.0 * train_correct / train_total

        # Validation phase
        val_loss, val_acc = evaluate_model(model, val_loader, device, criterion)

        epoch_time = time.time() - epoch_start

        # Update history
        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)
        history["epoch_time"].append(epoch_time)

        # Track best performance
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            history["best_epoch"] = epoch
            history["best_val_acc"] = val_acc
            patience_counter = 0
        else:
            patience_counter += 1

        # Track convergence (first time reaching 95% accuracy)
        if history["convergence_epoch"] is None and val_acc >= 95.0:
            history["convergence_epoch"] = epoch

        # Logging
        if epoch % log_interval == 0 or epoch == num_epochs - 1:
            print(
                f"Epoch {epoch}/{num_epochs} | "
                f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}% | "
                f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.2f}% | "
                f"Time: {epoch_time:.2f}s"
            )

        # Early stopping
        if early_stopping_patience is not None and patience_counter >= early_stopping_patience:
            print(f"Early stopping at epoch {epoch} (no improvement for {early_stopping_patience} epochs)")
            break

    return history


def evaluate_model(
    model: nn.Module,
    data_loader: DataLoader,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
    criterion: Optional[nn.Module] = None,
) -> Tuple[float, float]:
    """
    Evaluate model on a dataset.

    Args:
        model: Neural network model
        data_loader: Data loader for evaluation
        device: Device to evaluate on
        criterion: Loss function (default: CrossEntropyLoss)

    Returns:
        Tuple of (average_loss, accuracy_percentage)
    """
    model.eval()

    if criterion is None:
        criterion = nn.CrossEntropyLoss()

    total_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for inputs, targets in data_loader:
            inputs, targets = inputs.to(device), targets.to(device)

            outputs = model(inputs)
            loss = criterion(outputs, targets)

            total_loss += loss.item() * inputs.size(0)
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

    avg_loss = total_loss / total
    accuracy = 100.0 * correct / total

    return avg_loss, accuracy


def compute_convergence_metrics(history: Dict[str, List[float]]) -> Dict[str, float]:
    """
    Compute convergence metrics from training history.

    Args:
        history: Training history dictionary

    Returns:
        Dictionary with convergence metrics:
            - steps_to_90: Epochs to reach 90% val accuracy
            - steps_to_95: Epochs to reach 95% val accuracy
            - steps_to_99: Epochs to reach 99% val accuracy
            - final_val_acc: Final validation accuracy
            - best_val_acc: Best validation accuracy
            - train_test_gap: Gap between train and test accuracy
    """
    val_acc = history["val_acc"]
    train_acc = history["train_acc"]

    metrics = {
        "steps_to_90": None,
        "steps_to_95": None,
        "steps_to_99": None,
        "final_val_acc": val_acc[-1] if val_acc else 0.0,
        "best_val_acc": history.get("best_val_acc", max(val_acc) if val_acc else 0.0),
        "train_test_gap": train_acc[-1] - val_acc[-1] if (train_acc and val_acc) else 0.0,
    }

    # Find first epoch reaching accuracy thresholds
    for i, acc in enumerate(val_acc):
        if metrics["steps_to_90"] is None and acc >= 90.0:
            metrics["steps_to_90"] = i
        if metrics["steps_to_95"] is None and acc >= 95.0:
            metrics["steps_to_95"] = i
        if metrics["steps_to_99"] is None and acc >= 99.0:
            metrics["steps_to_99"] = i

    return metrics
