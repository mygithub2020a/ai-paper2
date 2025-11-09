import json

optimizers = ["BelOptim", "BelOptimWithMomentum", "Adam", "SGD"]
summary = {}

print("Benchmark Results Summary:")
print("--------------------------")
print(f"Task: Modular Addition (p=113), Epochs: 10")
print("--------------------------")
print(f"{'Optimizer':<25} | {'Final Train Loss':<20} | {'Final Val Accuracy (%)':<25}")
print("-" * 75)

for optimizer in optimizers:
    try:
        with open(f"results_{optimizer}.json", 'r') as f:
            data = json.load(f)
            final_loss = data['history']['train_loss'][-1]
            final_accuracy = data['history']['val_accuracy'][-1]
            summary[optimizer] = {"final_loss": final_loss, "final_accuracy": final_accuracy}
            print(f"{optimizer:<25} | {final_loss:<20.4f} | {final_accuracy:<25.2f}")
    except FileNotFoundError:
        print(f"Results for {optimizer} not found.")

print("-" * 75)
