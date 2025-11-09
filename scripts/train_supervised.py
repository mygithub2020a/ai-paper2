import torch
import torch.nn as nn
from belopt.optim import BelOpt
from belopt.models import SimpleMLP
from data.mod_arith import modular_addition_data
from belopt.schedules import ConstantScheduler, CosineDecayScheduler
import sys
import os

# Add the project root to the python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

def main():
    # Parameters
    p = 97
    n_samples = 1000
    input_dim = 2
    hidden_dim = 64
    output_dim = p  # For classification, output one logit per class
    epochs = 100
    batch_size = 64
    grad_clip = 1.0

    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Data
    x, y = modular_addition_data(p, n_samples)
    x = x.float().to(device)
    y = y.long().to(device) # CrossEntropyLoss expects long tensors for targets
    dataset = torch.utils.data.TensorDataset(x, y)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)

    total_steps = len(dataloader) * epochs

    # Schedulers
    eta_scheduler = CosineDecayScheduler(1e-3, total_steps)
    gamma_scheduler = ConstantScheduler(1e-4)
    beta_scheduler = CosineDecayScheduler(1e-3, total_steps)

    # Model, Optimizer, Loss
    model = SimpleMLP(input_dim, hidden_dim, output_dim).to(device)
    optimizer = BelOpt(model.parameters(),
                       eta_scheduler=eta_scheduler,
                       gamma_scheduler=gamma_scheduler,
                       beta_scheduler=beta_scheduler)
    loss_fn = nn.CrossEntropyLoss()

    # Training loop
    for epoch in range(epochs):
        total_loss = 0
        for x_batch, y_batch in dataloader:
            optimizer.zero_grad()

            # Forward pass
            logits = model(x_batch)
            loss = loss_fn(logits, y_batch)

            # Backward pass
            loss.backward()

            # Gradient clipping
            if grad_clip is not None:
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)

            # Optimization
            optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / len(dataloader)
        if (epoch + 1) % 10 == 0:
            print(f"Epoch [{epoch+1}/{epochs}], Loss: {avg_loss:.4f}")

if __name__ == '__main__':
    main()
