import os
import json

def analyze_results(log_dir='logs'):
    best_accuracy = 0
    best_config = None

    for filename in os.listdir(log_dir):
        if filename.endswith('.json'):
            filepath = os.path.join(log_dir, filename)
            with open(filepath, 'r') as f:
                try:
                    results = json.load(f)
                    max_accuracy = max(r['accuracy'] for r in results)
                    if max_accuracy > best_accuracy:
                        best_accuracy = max_accuracy
                        best_config = filename
                except (json.JSONDecodeError, ValueError) as e:
                    print(f"Skipping malformed file {filename}: {e}")
                    continue

    print(f"Best accuracy: {best_accuracy:.4f}")
    print(f"Best configuration: {best_config}")

if __name__ == '__main__':
    analyze_results()
