import torch
import torch.nn as nn
import torch.optim as optim
from pettingzoo.classic import chess_v6
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
        self.fc1 = nn.Linear(input_size, 512)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(512, 256)
        self.relu2 = nn.ReLU()
        self.fc3 = nn.Linear(256, output_size)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        x = self.relu1(self.fc1(x))
        x = self.relu2(self.fc2(x))
        x = self.fc3(x)
        return self.softmax(x)

def train_chess(policy, optimizer, n_episodes=1000):
    rewards_history = []
    env = chess_v6.env()

    for i_episode in range(n_episodes):
        env.reset()
        saved_log_probs = {agent: [] for agent in env.agents}
        rewards = {agent: [] for agent in env.agents}

        for agent in env.agent_iter():
            observation, reward, termination, truncation, info = env.last()

            if termination or truncation:
                break

            obs_tensor = torch.from_numpy(observation['observation'].copy()).float().flatten().unsqueeze(0)
            action_mask = observation['action_mask']

            probs = policy(obs_tensor)
            probs = probs * torch.from_numpy(action_mask).float()

            if torch.sum(probs) > 0:
                action = torch.multinomial(probs, 1).item()
            else:
                action = np.random.choice(np.where(action_mask == 1)[0])

            log_prob = torch.log(probs.squeeze(0)[action] + 1e-9)
            saved_log_probs[agent].append(log_prob)

            env.step(action)

        # Accumulate rewards
        for agent in env.agents:
            total_reward = env.rewards.get(agent, 0)
            rewards[agent].append(total_reward)

        # Update policy
        all_policy_losses = []
        for agent in env.agents:
            if not saved_log_probs[agent]:
                continue

            R = 0
            policy_loss = []
            returns = []

            # Calculate returns
            for r in rewards[agent][::-1]:
                R = r + 0.99 * R
                returns.insert(0, R)

            returns = torch.tensor(returns, dtype=torch.float32)
            if len(returns) > 1:
                returns = (returns - returns.mean()) / (returns.std() + 1e-9)

            for log_prob, R in zip(saved_log_probs[agent], returns):
                policy_loss.append(-log_prob * R)

            if policy_loss:
                all_policy_losses.extend(policy_loss)

        if all_policy_losses:
            optimizer.zero_grad()
            policy_loss_tensor = torch.stack(all_policy_losses).sum()
            policy_loss_tensor.backward()
            optimizer.step()

        rewards_history.append(np.sum([sum(v) for k, v in rewards.items()]))
        if (i_episode + 1) % 100 == 0:
            print(f'Episode {i_episode+1}/{n_episodes}, Last Reward: {rewards_history[-1]:.2f}')

    return rewards_history

if __name__ == '__main__':
    env = chess_v6.env()
    env.reset()
    observation, _, _, _, _ = env.last()
    input_size = observation['observation'].flatten().shape[0]
    output_size = env.action_space(env.agents[0]).n

    # Test with Belavkin Optimizer
    policy_belavkin = Policy(input_size, output_size)
    optimizer_belavkin = BelavkinOptimizer(policy_belavkin.parameters(), eta=1e-4, gamma=1e-5, beta=1e-3, clip_value=1.0)
    print("--- Running with Belavkin Optimizer ---")
    rewards_belavkin = train_chess(policy_belavkin, optimizer_belavkin)

    # Test with Adam Optimizer for comparison
    policy_adam = Policy(input_size, output_size)
    optimizer_adam = optim.Adam(policy_adam.parameters(), lr=1e-3)
    print("\n--- Running with Adam Optimizer ---")
    rewards_adam = train_chess(policy_adam, optimizer_adam)

    plt.figure(figsize=(12, 6))
    plt.plot(rewards_belavkin, label='Belavkin Optimizer')
    plt.plot(rewards_adam, label='Adam Optimizer')
    plt.title('Optimizer Performance on Chess-v6')
    plt.xlabel('Episode')
    plt.ylabel('Total Reward')
    plt.legend()
    plt.grid(True)
    plt.savefig('chess_benchmark.png')
    plt.show()

    env.close()
