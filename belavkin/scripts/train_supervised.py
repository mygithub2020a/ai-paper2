import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from belavkin.belopt.optim import BelOpt
from belavkin.data.mod_arith import generate_modular_arithmetic_data
from belavkin.data.mod_comp import generate_modular_composition_data

class MLP(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dim=128, num_layers=2):
        super(MLP, self).__init__()
        layers = []
        layers.append(nn.Linear(input_dim, hidden_dim))
        layers.append(nn.ReLU())
        for _ in range(num_layers - 1):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            layers.append(nn.ReLU())
        layers.append(nn.Linear(hidden_dim, output_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)

def train(args):
    # Setup device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load data
    if args.dataset == 'mod_arith':
        X, y = generate_modular_arithmetic_data(args.task, args.p, args.dim, args.num_samples)
        input_dim = X.shape[1]
        output_dim = y.shape[1]
    elif args.dataset == 'mod_comp':
        X, y = generate_modular_composition_data(args.p, args.dim, args.depth, args.num_samples)
        input_dim = X.shape[1]
        output_dim = y.shape[1]
    else:
        raise ValueError(f"Unknown dataset: {args.dataset}")

    X, y = X.to(device), y.to(device)

    # Create model
    model = MLP(input_dim, output_dim).to(device)

    # Create optimizer
    if args.optimizer == 'belopt':
        optimizer = BelOpt(model.parameters(), lr=args.lr, gamma0=args.gamma0, beta0=args.beta0)
    elif args.optimizer == 'adam':
        optimizer = optim.Adam(model.parameters(), lr=args.lr)
    elif args.optimizer == 'sgd':
        optimizer = optim.SGD(model.parameters(), lr=args.lr)
    elif args.optimizer == 'rmsprop':
        optimizer = optim.RMSprop(model.parameters(), lr=args.lr)
    else:
        raise ValueError(f"Unknown optimizer: {args.optimizer}")

    # Loss function
    criterion = nn.MSELoss()

    # Training loop
    for epoch in range(args.epochs):
        optimizer.zero_grad()
        outputs = model(X)
        loss = criterion(outputs, y)
        loss.backward()
        optimizer.step()

        if (epoch + 1) % 100 == 0:
            print(f'Epoch [{epoch+1}/{args.epochs}], Loss: {loss.item():.4f}')

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Belavkin Optimizer Supervised Training')
    parser.add_argument('--dataset', type=str, default='mod_arith', choices=['mod_arith', 'mod_comp'])
    parser.add_argument('--task', type=str, default='add', choices=['add', 'mul', 'inv', 'pow'])
    parser.add_argument('--p', type=int, default=97, help='modulus')
    parser.add_argument('--dim', type=int, default=1, help='input dimension')
    parser.add_argument('--depth', type=int, default=2, help='composition depth for mod_comp')
    parser.add_argument('--num_samples', type=int, default=1000)
    parser.add_argument('--optimizer', type=str, default='belopt', choices=['belopt', 'adam', 'sgd', 'rmsprop'])
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--gamma0', type=float, default=0.0)
    parser.add_argument('--beta0', type=float, default=0.0)
    parser.add_argument('--epochs', type=int, default=1000)
    args = parser.parse_args()
    train(args)
