import torch
import matplotlib.pyplot as plt
from belavkin.optimizer.optimizer import BelavkinOptimizer
from belavkin.benchmarks.synthetic import NonMarkovianQuadratic

def run_benchmark(optimizer_class, model, n_steps=200, **kwargs):
    optimizer = optimizer_class(model.parameters(), **kwargs)
    losses = []
    for _ in range(n_steps):
        optimizer.zero_grad()
        loss = model()
        loss.backward()
        optimizer.step()
        losses.append(loss.item())
    return losses

# --- Setup ---
torch.manual_seed(42)
n_steps = 200

configurations = {
    "Belavkin (Full)": (BelavkinOptimizer, {"lr": 0.1, "gamma": 0.1, "eta": 0.1, "beta": 0.01}),
    "No Adaptive Damping (gamma=0)": (BelavkinOptimizer, {"lr": 0.1, "gamma": 0, "eta": 0.1, "beta": 0.01}),
    "No Nonlinear Collapse (eta=0)": (BelavkinOptimizer, {"lr": 0.1, "gamma": 0.1, "eta": 0, "beta": 0.01}),
    "No Stochastic Exploration (beta=0)": (BelavkinOptimizer, {"lr": 0.1, "gamma": 0.1, "eta": 0.1, "beta": 0}),
    "Adam (Baseline)": (torch.optim.Adam, {"lr": 0.1}),
}

results = {}
plt.figure(figsize=(12, 8))

for name, (optimizer_class, kwargs) in configurations.items():
    print(f"Running benchmark for: {name}")
    model = NonMarkovianQuadratic()
    losses = run_benchmark(optimizer_class, model, n_steps=n_steps, **kwargs)
    results[name] = losses[-1]
    plt.plot(losses, label=name)

# --- Plotting ---
plt.xlabel('Steps')
plt.ylabel('Loss')
plt.title('Ablation Study on Non-Markovian Quadratic Problem')
plt.legend()
plt.grid(True)
plt.savefig('belavkin/benchmarks/ablation_study.png')
print("\\nAblation study complete. Plot saved to belavkin/benchmarks/ablation_study.png")

# --- Print Results Table ---
print("\\n--- Final Loss Results ---")
print("| Optimizer Configuration          | Final Loss |")
print("|----------------------------------|------------|")
for name, final_loss in results.items():
    print(f"| {name:<32} | {final_loss:.6f} |")
