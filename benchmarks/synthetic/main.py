import torch
import torch.nn as nn
import torch.optim as optim
import argparse
import json
import os
import sys

# Add the root directory to the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from benchmarks.synthetic.modular_arithmetic import create_modular_arithmetic_dataset
from benchmarks.synthetic.model import MLP
from belavkin.belopt.optim import BelavkinOptimizer

def main():
    parser = argparse.ArgumentParser(description='Belavkin Optimizer Benchmark')
    parser.add_argument('--optimizer', type=str, default='belavkin',
                        help='Optimizer to use (belavkin, adam, sgd)')
    parser.add_argument('--lr', type=float, default=1e-3,
                        help='Learning rate')
    parser.add_argument('--gamma', type=float, default=1e-4,
                        help='Gamma for Belavkin Optimizer')
    parser.add_argument('--beta', type=float, default=1e-2,
                        help='Beta for Belavkin Optimizer')
    parser.add_argument('--adaptive_gamma', action='store_true',
                        help='Use adaptive gamma for Belavkin Optimizer')
    parser.add_argument('--epochs', type=int, default=100,
                        help='Number of training epochs')
    parser.add_argument('--p', type=int, default=97,
                        help='Prime modulus for modular arithmetic')
    parser.add_argument('--a', type=int, default=3,
                        help='Multiplier for modular arithmetic')
    parser.add_argument('--b', type=int, default=5,
                        help='Offset for modular arithmetic')
    parser.add_argument('--hidden_size', type=int, default=128,
                        help='Hidden size of the MLP')
    parser.add_argument('--embedding_dim', type=int, default=32,
                        help='Embedding dimension for the input')
    parser.add_argument('--log_dir', type=str, default='logs',
                        help='Directory to save logs')
    args = parser.parse_args()

    # Create dataset
    x_train, y_train = create_modular_arithmetic_dataset(args.p, args.a, args.b)
    x_test, y_test = create_modular_arithmetic_dataset(args.p, args.a, args.b)

    # Create model
    model = MLP(args.p, args.hidden_size, args.p, args.embedding_dim)

    # Create optimizer
    if args.optimizer.lower() == 'belavkin':
        optimizer = BelavkinOptimizer(model.parameters(), lr=args.lr, gamma=args.gamma, beta=args.beta, adaptive_gamma=args.adaptive_gamma)
    elif args.optimizer.lower() == 'adam':
        optimizer = optim.Adam(model.parameters(), lr=args.lr)
    elif args.optimizer.lower() == 'sgd':
        optimizer = optim.SGD(model.parameters(), lr=args.lr)
    else:
        raise ValueError("Unknown optimizer: {}".format(args.optimizer))

    criterion = nn.CrossEntropyLoss()

    # Training loop
    results = []
    for epoch in range(args.epochs):
        model.train()
        optimizer.zero_grad()
        output = model(x_train)
        loss = criterion(output, y_train.squeeze())
        loss.backward()
        optimizer.step()

        # Evaluation
        model.eval()
        with torch.no_grad():
            test_output = model(x_test)
            test_loss = criterion(test_output, y_test.squeeze())
            _, predicted = torch.max(test_output, 1)
            accuracy = (predicted == y_test.squeeze()).float().mean()

        results.append({
            'epoch': epoch,
            'train_loss': loss.item(),
            'test_loss': test_loss.item(),
            'accuracy': accuracy.item()
        })
        print(f'Epoch {epoch+1}/{args.epochs}, Train Loss: {loss.item():.4f}, Test Loss: {test_loss.item():.4f}, Accuracy: {accuracy.item():.4f}')

    # Save results
    if not os.path.exists(args.log_dir):
        os.makedirs(args.log_dir)

    log_file = os.path.join(args.log_dir, f'{args.optimizer}_lr{args.lr}_gamma{args.gamma}_beta{args.beta}_embedding.json')
    with open(log_file, 'w') as f:
        json.dump(results, f, indent=4)

if __name__ == '__main__':
    main()
