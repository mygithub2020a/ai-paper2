import os
import json
import pandas as pd

def analyze_all_results(log_dir='logs'):
    results = []
    for filename in os.listdir(log_dir):
        if filename.endswith('.json'):
            filepath = os.path.join(log_dir, filename)
            with open(filepath, 'r') as f:
                try:
                    data = json.load(f)
                    final_accuracy = data[-1]['accuracy']
                    results.append({'config': filename, 'final_accuracy': final_accuracy})
                except (json.JSONDecodeError, ValueError, IndexError) as e:
                    print(f"Skipping malformed or empty file {filename}: {e}")
                    continue

    df = pd.DataFrame(results)
    df = df.sort_values(by='final_accuracy', ascending=False)
    print(df)

if __name__ == '__main__':
    analyze_all_results()
