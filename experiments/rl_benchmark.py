
import json
import numpy as np
import argparse
from belavkin.rl.ppo_ig import train_ppo

def run_rl_experiments(steps=500):
    results = {}

    print(f"Running Baseline PPO ({steps} steps)...")
    rew_base, innov_base = train_ppo(use_ig=False, steps=steps)
    results['PPO'] = {'rewards': rew_base, 'innovation': innov_base}

    print(f"Running IG-PPO ({steps} steps)...")
    rew_ig, innov_ig = train_ppo(use_ig=True, steps=steps)
    results['IG-PPO'] = {'rewards': rew_ig, 'innovation': innov_ig}

    # Ablation 1: Random Signal (Simulated by passing random flag, but for now reusing the same func)
    # We need to modify ppo_ig to accept a 'signal_type' arg to do this properly.
    # For now, we'll stick to the main comparison.

    return results

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--steps', type=int, default=200)
    parser.add_argument('--out', type=str, default='results_rl.json')
    args = parser.parse_args()

    res = run_rl_experiments(args.steps)

    # Summary
    print("--- RL Results ---")
    print(f"PPO Final Reward: {np.mean(res['PPO']['rewards'][-10:]):.2f}")
    print(f"IG-PPO Final Reward: {np.mean(res['IG-PPO']['rewards'][-10:]):.2f}")

    with open(args.out, 'w') as f:
        json.dump(res, f)
