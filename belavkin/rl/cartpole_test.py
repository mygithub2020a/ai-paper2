import torch
import torch.nn as nn
import torch.optim as optim
import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt

# Custom optimizer
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from optimizer.belavkin_optimizer import BelavkinOptimizer

# Define the policy network
class Policy(nn.Module):
    def __init__(self, input_size, output_size):
        super(Policy, self).__init__()
        self.fc1 = nn.Linear(input_size, 128)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(128, output_size)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        x = self.fc2(self.relu(self.fc1(x)))
        return self.softmax(x)

# REINFORCE algorithm
def reinforce(policy, optimizer, n_episodes=1000, gamma=0.99):
    rewards_history = []
    for i_episode in range(n_episodes):
        saved_log_probs = []
        rewards = []
        state, _ = env.reset()

        for t in range(1000): # Don't infinite loop
            state_tensor = torch.from_numpy(state).float().unsqueeze(0)
            probs = policy(state_tensor)
            action = torch.multinomial(probs, 1).item()
            saved_log_probs.append(torch.log(probs.squeeze(0)[action]))

            state, reward, done, _, _ = env.step(action)
            rewards.append(reward)
            if done:
                break

        R = 0
        policy_loss = []
        returns = []
        for r in rewards[::-1]:
            R = r + gamma * R
            returns.insert(0, R)

        returns = torch.tensor(returns)
        returns = (returns - returns.mean()) / (returns.std() + 1e-9)

        for log_prob, R in zip(saved_log_probs, returns):
            policy_loss.append(-log_prob * R)

        optimizer.zero_grad()
        policy_loss = torch.stack(policy_loss).sum()
        policy_loss.backward()
        optimizer.step()

        rewards_history.append(np.sum(rewards))
        if (i_episode + 1) % 100 == 0:
            print(f'Episode {i_episode+1}/{n_episodes}, Last Reward: {np.sum(rewards):.2f}')

    return rewards_history

if __name__ == '__main__':
    env = gym.make('CartPole-v1')

    input_size = env.observation_space.shape[0]
    output_size = env.action_space.n

    # Test with Belavkin Optimizer
    policy_belavkin = Policy(input_size, output_size)
    optimizer_belavkin = BelavkinOptimizer(policy_belavkin.parameters(), eta=1e-3, gamma=1e-4, beta=1e-2, clip_value=1.0)
    print("--- Running with Belavkin Optimizer ---")
    rewards_belavkin = reinforce(policy_belavkin, optimizer_belavkin)

    # Test with Adam Optimizer for comparison
    policy_adam = Policy(input_size, output_size)
    optimizer_adam = optim.Adam(policy_adam.parameters(), lr=1e-2)
    print("\n--- Running with Adam Optimizer ---")
    rewards_adam = reinforce(policy_adam, optimizer_adam)

    plt.figure(figsize=(12, 6))
    plt.plot(rewards_belavkin, label='Belavkin Optimizer')
    plt.plot(rewards_adam, label='Adam Optimizer')
    plt.title('Optimizer Performance on CartPole-v1')
    plt.xlabel('Episode')
    plt.ylabel('Total Reward')
    plt.legend()
    plt.grid(True)
    plt.savefig('cartpole_benchmark.png')
    plt.show()

    env.close()
