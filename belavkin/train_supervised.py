import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import argparse
import os

# --- Import local modules ---
# Note: This script assumes it's run from the root of the project,
# and the 'belavkin' directory is in the Python path.
from optim.belavkin import Belavkin
from datasets.mod_arith import ModularArithmeticDataset
from datasets.mod_comp import ModularCompositionDataset
from models.mlp import MLP
from models.transformer import TransformerModel

def get_optimizer(optimizer_name, model_params, args):
    """Instantiates the optimizer based on the provided name and arguments."""
    if optimizer_name == 'belavkin':
        return Belavkin(model_params, lr=args.lr, eta=args.eta, beta=args.beta,
                        adaptive_gamma=not args.no_adaptive_gamma)
    elif optimizer_name == 'adam':
        return torch.optim.Adam(model_params, lr=args.lr)
    elif optimizer_name == 'sgd':
        return torch.optim.SGD(model_params, lr=args.lr, momentum=0.9)
    elif optimizer_name == 'rmsprop':
        return torch.optim.RMSprop(model_params, lr=args.lr)
    else:
        raise ValueError(f"Unknown optimizer: {optimizer_name}")

def train(args):
    """Main training loop."""
    # --- Setup ---
    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")

    # --- Data ---
    if args.task == 'mod_arith':
        train_dataset = ModularArithmeticDataset(args.num_train_samples, args.modulus, seed=args.seed)
        val_dataset = ModularArithmeticDataset(args.num_val_samples, args.modulus, seed=args.seed + 1,
                                               a=train_dataset.a, b=train_dataset.b)
    elif args.task == 'mod_comp':
        train_dataset = ModularCompositionDataset(args.num_train_samples, args.modulus, args.num_functions, seed=args.seed)
        val_dataset = ModularCompositionDataset(args.num_val_samples, args.modulus, args.num_functions, seed=args.seed + 1)
    else:
        raise ValueError(f"Unknown task: {args.task}")

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size)

    # --- Model ---
    if args.model == 'mlp':
        model = MLP(args.modulus, args.embed_dim, args.hidden_dim, args.num_layers, args.modulus)
    elif args.model == 'transformer':
        model = TransformerModel(args.modulus, args.d_model, args.nhead, args.num_layers,
                                 args.dim_feedforward, args.modulus)
    else:
        raise ValueError(f"Unknown model: {args.model}")
    model.to(device)

    # --- Optimizer and Loss ---
    optimizer = get_optimizer(args.optimizer, model.parameters(), args)
    criterion = nn.CrossEntropyLoss()

    # --- Logging ---
    log_dir = os.path.join(args.log_dir, f"{args.task}_{args.model}_{args.optimizer}_lr{args.lr}_seed{args.seed}")
    writer = SummaryWriter(log_dir)

    # --- Training Loop ---
    print(f"Starting training for {args.epochs} epochs on device {device}")
    global_step = 0
    for epoch in range(args.epochs):
        model.train()
        for batch in train_loader:
            optimizer.zero_grad()

            # Move data to device and handle different data formats
            if args.task == 'mod_arith':
                inputs, targets = batch
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = model(inputs)
            elif args.task == 'mod_comp':
                inputs, funcs, targets = batch
                inputs, funcs, targets = inputs.to(device), funcs.to(device), targets.to(device)
                outputs = model(inputs, funcs)

            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

            if global_step % 100 == 0:
                writer.add_scalar('Loss/train', loss.item(), global_step)
            global_step += 1

        # --- Validation Loop ---
        model.eval()
        total_val_loss = 0
        correct = 0
        total = 0
        with torch.no_grad():
            for batch in val_loader:
                if args.task == 'mod_arith':
                    inputs, targets = batch
                    inputs, targets = inputs.to(device), targets.to(device)
                    outputs = model(inputs)
                elif args.task == 'mod_comp':
                    inputs, funcs, targets = batch
                    inputs, funcs, targets = inputs.to(device), funcs.to(device), targets.to(device)
                    outputs = model(inputs, funcs)

                loss = criterion(outputs, targets)
                total_val_loss += loss.item()

                _, predicted = torch.max(outputs.data, 1)
                total += targets.size(0)
                correct += (predicted == targets).sum().item()

        avg_val_loss = total_val_loss / len(val_loader)
        val_accuracy = 100 * correct / total

        writer.add_scalar('Loss/validation', avg_val_loss, epoch)
        writer.add_scalar('Accuracy/validation', val_accuracy, epoch)

        print(f"Epoch {epoch+1}/{args.epochs} | Val Loss: {avg_val_loss:.4f} | Val Acc: {val_accuracy:.2f}%")

    writer.close()
    print("Training complete.")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Train models for supervised learning tasks.")
    # Task and Model
    parser.add_argument('--task', type=str, required=True, choices=['mod_arith', 'mod_comp'], help='Task to train on.')
    parser.add_argument('--model', type=str, required=True, choices=['mlp', 'transformer'], help='Model to use.')
    parser.add_argument('--modulus', type=int, default=97, help='Modulus for the task.')

    # Training
    parser.add_argument('--epochs', type=int, default=10, help='Number of training epochs.')
    parser.add_argument('--batch_size', type=int, default=256, help='Batch size.')
    parser.add_argument('--seed', type=int, default=42, help='Random seed.')
    parser.add_argument('--no_cuda', action='store_true', help='Disable CUDA training.')
    parser.add_argument('--log_dir', type=str, default='runs', help='Directory for TensorBoard logs.')
    parser.add_argument('--num_train_samples', type=int, default=100000)
    parser.add_argument('--num_val_samples', type=int, default=20000)

    # Optimizer
    parser.add_argument('--optimizer', type=str, default='belavkin', choices=['belavkin', 'adam', 'sgd', 'rmsprop'])
    parser.add_argument('--lr', type=float, default=1e-3, help='Learning rate.')
    # Belavkin specific
    parser.add_argument('--eta', type=float, default=1.0)
    parser.add_argument('--beta', type=float, default=0.1)
    parser.add_argument('--no_adaptive_gamma', action='store_true')

    # Model Hyperparameters
    # MLP
    parser.add_argument('--embed_dim', type=int, default=128)
    parser.add_argument('--hidden_dim', type=int, default=512)
    # Transformer
    parser.add_argument('--d_model', type=int, default=128)
    parser.add_argument('--nhead', type=int, default=4)
    parser.add_argument('--dim_feedforward', type=int, default=512)
    # Both
    parser.add_argument('--num_layers', type=int, default=4)
    # Mod Comp specific
    parser.add_argument('--num_functions', type=int, default=4)

    args = parser.parse_args()
    train(args)
