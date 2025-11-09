import argparse
import torch
from beloptim.benchmarks.training import train_modular_arithmetic
import json

def main():
    parser = argparse.ArgumentParser(description="Run synthetic modular arithmetic benchmarks.")
    parser.add_argument('--optimizer', type=str, required=True,
                        choices=['BelOptim', 'BelOptimWithMomentum', 'BelOptimAdaptive',
                                 'BelOptimLayerwise', 'Adam', 'SGD'],
                        help="Optimizer to use.")
    parser.add_argument('--p', type=int, required=True, help="Modulus for the arithmetic task.")
    parser.add_argument('--operation', type=str, default='add',
                        choices=['add', 'compose', 'poly'],
                        help="Modular arithmetic operation.")
    parser.add_argument('--d_model', type=int, default=128, help="Model dimension.")
    parser.add_argument('--n_layers', type=int, default=2, help="Number of transformer layers.")
    parser.add_argument('--n_head', type=int, default=8, help="Number of attention heads.")
    parser.add_argument('--num_samples', type=int, default=10000, help="Number of training samples.")
    parser.add_argument('--batch_size', type=int, default=64, help="Batch size.")
    parser.add_argument('--epochs', type=int, default=10, help="Number of training epochs.")
    parser.add_argument('--lr', type=float, default=1e-3, help="Learning rate.")
    parser.add_argument('--gamma', type=float, default=0.1, help="Gamma parameter for BelOptim.")
    parser.add_argument('--beta', type=float, default=0.01, help="Beta parameter for BelOptim.")
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu',
                        help="Device to use for training.")
    parser.add_argument('--output_file', type=str, default='results.json',
                        help="File to save the results.")

    args = parser.parse_args()

    print(f"Running benchmark with the following settings:\n{args}")

    history = train_modular_arithmetic(
        optimizer_name=args.optimizer,
        p=args.p,
        operation=args.operation,
        d_model=args.d_model,
        n_layers=args.n_layers,
        n_head=args.n_head,
        num_samples=args.num_samples,
        batch_size=args.batch_size,
        epochs=args.epochs,
        lr=args.lr,
        gamma=args.gamma,
        beta=args.beta,
        device=args.device
    )

    results = {
        'args': vars(args),
        'history': history
    }

    with open(args.output_file, 'w') as f:
        json.dump(results, f, indent=4)

    print(f"Results saved to {args.output_file}")

if __name__ == '__main__':
    main()
