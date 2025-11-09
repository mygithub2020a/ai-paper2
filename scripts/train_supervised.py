import torch
import torch.nn as nn
from belopt.optim import BelOpt
from belopt.models import SimpleMLP
from data.mod_arith import get_dataloader
from belopt.schedules import get_scheduler

def main():
    # --- Configuration ---
    p = 97
    op = 'add'
    batch_size = 128

    if op in ['add', 'multiply', 'power']:
        num_inputs = 2
    elif op == 'invert':
        num_inputs = 1
    else:
        raise ValueError(f"Unknown operation: {op}")

    input_dim = num_inputs * p
    hidden_dim = 128
    output_dim = p
    num_layers = 3

    learning_rate = 1e-3
    gamma = 1e-4
    beta = 1e-3
    epochs = 10

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # --- Data ---
    train_loader = get_dataloader(p, op, batch_size)

    # --- Model ---
    model = SimpleMLP(input_dim, hidden_dim, output_dim, num_layers).to(device)

    # --- Optimizer ---
    optimizer = BelOpt(model.parameters(), lr=learning_rate, gamma=gamma, beta=beta)

    # --- Schedulers ---
    lr_scheduler = get_scheduler(optimizer, 'lr', 'cosine', T_max=epochs)
    gamma_scheduler = get_scheduler(optimizer, 'gamma', 'cosine', T_max=epochs)
    beta_scheduler = get_scheduler(optimizer, 'beta', 'cosine', T_max=epochs)

    # --- Loss Function ---
    criterion = nn.CrossEntropyLoss()

    # --- Training Loop ---
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for x, y in train_loader:
            x = x.float().to(device)
            y = y.long().to(device)

            optimizer.zero_grad()

            # One-hot encode the input
            x_one_hot = torch.nn.functional.one_hot(x.long(), num_classes=p).view(x.size(0), -1).float()

            outputs = model(x_one_hot)
            loss = criterion(outputs, y)

            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / len(train_loader)
        print(f"Epoch [{epoch+1}/{epochs}], Loss: {avg_loss:.4f}")

        lr_scheduler.step()
        gamma_scheduler.step()
        beta_scheduler.step()


if __name__ == '__main__':
    main()
