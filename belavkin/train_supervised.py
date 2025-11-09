import argparse
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from torch.utils.tensorboard import SummaryWriter
import os
from datetime import datetime

from models.mlp import MLP
from optim.belavkin import Belavkin
from datasets.mod_arith import generate_mod_arith_dataset

def main(args):
    # --- 1. Setup & Reproducibility ---
    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Create a unique directory for this run
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    run_name = f"{args.run_name}_{timestamp}"
    log_dir = os.path.join(args.log_dir, run_name)
    writer = SummaryWriter(log_dir=log_dir)
    print(f"Logging to {log_dir}")

    # --- 2. Data Loading ---
    train_data, train_targets = generate_mod_arith_dataset(args.num_samples, args.modulus, seed=args.seed)
    val_data, val_targets = generate_mod_arith_dataset(int(args.num_samples * 0.2), args.modulus, seed=args.seed + 1)

    train_dataset = TensorDataset(train_data, train_targets)
    val_dataset = TensorDataset(val_data, val_targets)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size)

    # --- 3. Model, Optimizer, Loss ---
    model = MLP(
        vocab_size=args.modulus,
        embedding_dim=args.embedding_dim,
        hidden_dim=args.hidden_dim,
        output_dim=args.modulus,
        num_layers=args.num_layers
    ).to(device)

    optimizer = Belavkin(
        model.parameters(),
        lr=args.lr,
        eta=args.eta,
        beta=args.beta,
        gamma_init=args.gamma_init,
        adaptive_gamma=not args.no_adaptive_gamma
    )

    criterion = nn.CrossEntropyLoss()

    # --- 4. Training Loop ---
    print("Starting training...")
    global_step = 0
    for epoch in range(args.epochs):
        model.train()
        total_train_loss = 0

        for i, (inputs, targets) in enumerate(train_loader):
            inputs, targets = inputs.to(device), targets.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

            total_train_loss += loss.item()
            writer.add_scalar('Loss/train_step', loss.item(), global_step)
            global_step += 1

        avg_train_loss = total_train_loss / len(train_loader)
        writer.add_scalar('Loss/train_epoch', avg_train_loss, epoch)

        # --- 5. Validation Loop ---
        model.eval()
        total_val_loss = 0
        correct = 0
        total = 0
        with torch.no_grad():
            for inputs, targets in val_loader:
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                total_val_loss += loss.item()

                _, predicted = torch.max(outputs.data, 1)
                total += targets.size(0)
                correct += (predicted == targets).sum().item()

        avg_val_loss = total_val_loss / len(val_loader)
        accuracy = 100 * correct / total

        writer.add_scalar('Loss/validation', avg_val_loss, epoch)
        writer.add_scalar('Accuracy/validation', accuracy, epoch)

        print(f"Epoch [{epoch+1}/{args.epochs}], Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}, Val Acc: {accuracy:.2f}%")

    writer.close()
    print("Training finished.")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Train an MLP on the Modular Arithmetic task.")

    # Dataset args
    parser.add_argument('--modulus', type=int, default=97, help='Modulus for the arithmetic task')
    parser.add_argument('--num_samples', type=int, default=10000, help='Number of training samples')

    # Model args
    parser.add_argument('--embedding_dim', type=int, default=32, help='Dimension of embeddings')
    parser.add_argument('--hidden_dim', type=int, default=256, help='Dimension of hidden layers')
    parser.add_argument('--num_layers', type=int, default=4, help='Number of MLP layers')

    # Optimizer args
    parser.add_argument('--lr', type=float, default=1e-3, help='Learning rate')
    parser.add_argument('--eta', type=float, default=1.0, help='Optimizer eta parameter')
    parser.add_argument('--beta', type=float, default=0.1, help='Optimizer beta parameter')
    parser.add_argument('--gamma_init', type=float, default=0.1, help='Optimizer gamma_init parameter')
    parser.add_argument('--no_adaptive_gamma', action='store_true', help='Disable adaptive gamma in the optimizer')

    # Training args
    parser.add_argument('--epochs', type=int, default=10, help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=64, help='Batch size for training')
    parser.add_argument('--seed', type=int, default=42, help='Random seed for reproducibility')

    # Logging args
    parser.add_argument('--log_dir', type=str, default='runs', help='Directory for TensorBoard logs')
    parser.add_argument('--run_name', type=str, default='Belavkin_MLP', help='Name for the run')

    args = parser.parse_args()
    main(args)
