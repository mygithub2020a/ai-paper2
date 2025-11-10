"""
Main training script for supervised learning benchmarks.

Compares BelOpt with baseline optimizers (Adam, SGD, RMSProp) on modular arithmetic tasks.
"""

import argparse
import os
import sys
import time
from pathlib import Path

import torch
import torch.nn as nn
import yaml

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from belavkin.belopt import BelOpt
from belavkin.data import generate_modular_dataset
from belavkin.data.mod_comp import generate_composition_dataset
from belavkin.models import get_model
from belavkin.utils import (
    set_seed,
    get_device,
    count_parameters,
    AverageMeter,
    Timer,
    Logger,
    calculate_accuracy,
    print_metrics,
)


def train_epoch(model, train_loader, optimizer, criterion, device, modulus):
    """Train for one epoch."""
    model.train()
    loss_meter = AverageMeter()
    acc_meter = AverageMeter()

    for inputs, labels in train_loader:
        inputs = inputs.to(device)
        labels = labels.to(device).long()

        # Forward
        outputs = model(inputs)
        loss = criterion(outputs, labels)

        # Backward
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Metrics
        batch_size = inputs.size(0)
        loss_meter.update(loss.item(), batch_size)
        acc = calculate_accuracy(outputs, labels, modulus)
        acc_meter.update(acc, batch_size)

    return loss_meter.avg, acc_meter.avg


@torch.no_grad()
def evaluate(model, test_loader, criterion, device, modulus):
    """Evaluate on test set."""
    model.eval()
    loss_meter = AverageMeter()
    acc_meter = AverageMeter()

    for inputs, labels in test_loader:
        inputs = inputs.to(device)
        labels = labels.to(device).long()

        outputs = model(inputs)
        loss = criterion(outputs, labels)

        batch_size = inputs.size(0)
        loss_meter.update(loss.item(), batch_size)
        acc = calculate_accuracy(outputs, labels, modulus)
        acc_meter.update(acc, batch_size)

    return loss_meter.avg, acc_meter.avg


def get_optimizer(name, model_params, lr, **kwargs):
    """Get optimizer by name."""
    if name.lower() == 'belopt':
        return BelOpt(
            model_params,
            lr=lr,
            gamma0=kwargs.get('gamma0', 1e-3),
            beta0=kwargs.get('beta0', 0.0),
            deterministic=kwargs.get('deterministic', False),
            adaptive_gamma=kwargs.get('adaptive_gamma', True),
        )
    elif name.lower() == 'adam':
        return torch.optim.Adam(model_params, lr=lr)
    elif name.lower() == 'sgd':
        momentum = kwargs.get('momentum', 0.9)
        return torch.optim.SGD(model_params, lr=lr, momentum=momentum)
    elif name.lower() == 'rmsprop':
        return torch.optim.RMSprop(model_params, lr=lr)
    else:
        raise ValueError(f"Unknown optimizer: {name}")


def main(args):
    """Main training function."""
    # Set seed
    set_seed(args.seed)

    # Device
    device = get_device()
    print(f"Using device: {device}")

    # Create datasets
    print(f"\nCreating dataset: task={args.task}, modulus={args.modulus}, input_dim={args.input_dim}")
    if args.task in ['add', 'mul', 'inv', 'pow']:
        train_loader, test_loader = generate_modular_dataset(
            task=args.task,
            n_samples=args.n_train,
            modulus=args.modulus,
            input_dim=args.input_dim,
            batch_size=args.batch_size,
            seed=args.seed,
        )
        # Input size is 2*input_dim for binary ops, input_dim for unary
        if args.task in ['add', 'mul', 'pow']:
            input_size = 2 * args.input_dim + (1 if args.task == 'pow' else 0)
        else:
            input_size = args.input_dim
    elif args.task == 'composition':
        train_loader, test_loader = generate_composition_dataset(
            n_samples=args.n_train,
            modulus=args.modulus,
            input_dim=args.input_dim,
            composition_depth=args.composition_depth,
            batch_size=args.batch_size,
            seed=args.seed,
        )
        input_size = args.input_dim
    else:
        raise ValueError(f"Unknown task: {args.task}")

    print(f"Train batches: {len(train_loader)}, Test batches: {len(test_loader)}")

    # Create model
    print(f"\nCreating model: {args.model}")
    model = get_model(
        model_type=args.model,
        input_dim=input_size,
        output_dim=args.modulus,  # Classification over modulus
        hidden_dim=args.hidden_dim,
        n_layers=args.n_layers,
    ).to(device)

    n_params = count_parameters(model)
    print(f"Model parameters: {n_params:,}")

    # Create optimizer
    print(f"\nOptimizer: {args.optimizer}, LR: {args.lr}")
    optimizer = get_optimizer(
        args.optimizer,
        model.parameters(),
        lr=args.lr,
        gamma0=args.gamma0,
        beta0=args.beta0,
        deterministic=args.deterministic,
        momentum=args.momentum,
    )

    # Loss function
    criterion = nn.CrossEntropyLoss()

    # Logger
    exp_name = f"{args.task}_{args.optimizer}_m{args.modulus}_d{args.input_dim}_s{args.seed}"
    logger = Logger(args.log_dir, exp_name)

    # Training loop
    print(f"\nTraining for {args.epochs} epochs...\n")
    timer = Timer()
    best_test_acc = 0.0
    target_acc_time = None

    for epoch in range(1, args.epochs + 1):
        timer.start('epoch')

        # Train
        train_loss, train_acc = train_epoch(
            model, train_loader, optimizer, criterion, device, args.modulus
        )

        # Evaluate
        test_loss, test_acc = evaluate(
            model, test_loader, criterion, device, args.modulus
        )

        epoch_time = timer.stop('epoch')

        # Log
        logger.log({
            'epoch': epoch,
            'train_loss': train_loss,
            'train_acc': train_acc,
            'test_loss': test_loss,
            'test_acc': test_acc,
            'epoch_time': epoch_time,
        }, step=epoch)

        # Print
        if epoch % args.print_every == 0 or epoch == 1:
            print_metrics(epoch, train_loss, train_acc, test_loss, test_acc, epoch_time)

        # Track best
        if test_acc > best_test_acc:
            best_test_acc = test_acc

        # Track time to target accuracy
        if target_acc_time is None and test_acc >= args.target_acc:
            target_acc_time = sum(logger.logs[i]['epoch_time'] for i in range(epoch))

    # Final results
    print(f"\n{'='*60}")
    print(f"Training completed!")
    print(f"Best test accuracy: {best_test_acc:.2f}%")
    if target_acc_time is not None:
        print(f"Time to {args.target_acc}% accuracy: {target_acc_time:.2f}s")
    else:
        print(f"Did not reach {args.target_acc}% accuracy")
    print(f"Total training time: {sum(log['epoch_time'] for log in logger.logs):.2f}s")
    print(f"Logs saved to: {logger.log_file}")
    print(f"{'='*60}\n")

    return {
        'best_test_acc': best_test_acc,
        'final_test_acc': test_acc,
        'time_to_target': target_acc_time,
    }


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train on modular arithmetic tasks')

    # Task settings
    parser.add_argument('--task', type=str, default='add',
                        choices=['add', 'mul', 'inv', 'pow', 'composition'],
                        help='Task type')
    parser.add_argument('--modulus', type=int, default=97,
                        help='Prime modulus')
    parser.add_argument('--input_dim', type=int, default=1,
                        help='Input dimension')
    parser.add_argument('--composition_depth', type=int, default=2,
                        help='Composition depth (for composition task)')

    # Data settings
    parser.add_argument('--n_train', type=int, default=10000,
                        help='Number of training samples')
    parser.add_argument('--batch_size', type=int, default=64,
                        help='Batch size')

    # Model settings
    parser.add_argument('--model', type=str, default='mlp_small',
                        choices=['mlp_small', 'mlp_medium', 'mlp_large', 'residual', 'mixer'],
                        help='Model architecture')
    parser.add_argument('--hidden_dim', type=int, default=128,
                        help='Hidden dimension')
    parser.add_argument('--n_layers', type=int, default=4,
                        help='Number of layers')

    # Optimizer settings
    parser.add_argument('--optimizer', type=str, default='belopt',
                        choices=['belopt', 'adam', 'sgd', 'rmsprop'],
                        help='Optimizer')
    parser.add_argument('--lr', type=float, default=1e-3,
                        help='Learning rate')
    parser.add_argument('--gamma0', type=float, default=1e-3,
                        help='BelOpt: Initial gamma')
    parser.add_argument('--beta0', type=float, default=0.0,
                        help='BelOpt: Initial beta (exploration noise)')
    parser.add_argument('--deterministic', action='store_true',
                        help='BelOpt: Disable exploration noise')
    parser.add_argument('--momentum', type=float, default=0.9,
                        help='SGD: Momentum')

    # Training settings
    parser.add_argument('--epochs', type=int, default=100,
                        help='Number of epochs')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed')
    parser.add_argument('--target_acc', type=float, default=95.0,
                        help='Target accuracy for time-to-target metric')

    # Logging
    parser.add_argument('--log_dir', type=str, default='./results/supervised',
                        help='Log directory')
    parser.add_argument('--print_every', type=int, default=10,
                        help='Print frequency')

    args = parser.parse_args()
    main(args)
