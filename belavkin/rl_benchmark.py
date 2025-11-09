import gymnasium as gym
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical
import numpy as np

from belavkin.optimizer import BelavkinOptimizer

# Policy network
class Policy(nn.Module):
    def __init__(self, state_size, action_size):
        super(Policy, self).__init__()
        self.fc1 = nn.Linear(state_size, 128)
        self.fc2 = nn.Linear(128, action_size)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return F.softmax(x, dim=1)

def train(optimizer_class, **optimizer_kwargs):
    env = gym.make('CartPole-v1')
    state_size = env.observation_space.shape[0]
    action_size = env.action_space.n
    policy = Policy(state_size, action_size)
    optimizer = optimizer_class(policy.parameters(), **optimizer_kwargs)

    scores = []
    for i_episode in range(100):
        state, _ = env.reset()
        saved_log_probs = []
        rewards = []
        for t in range(1000):
            state = torch.from_numpy(state).float().unsqueeze(0)
            probs = policy(state)
            m = Categorical(probs)
            action = m.sample()
            saved_log_probs.append(m.log_prob(action))
            state, reward, done, _, _ = env.step(action.item())
            rewards.append(reward)
            if done:
                break

        scores.append(sum(rewards))

        R = 0
        policy_loss = []
        returns = []
        for r in rewards[::-1]:
            R = r + 0.99 * R
            returns.insert(0, R)
        returns = torch.tensor(returns)
        returns = (returns - returns.mean()) / (returns.std() + np.finfo(np.float32).eps.item())
        for log_prob, R in zip(saved_log_probs, returns):
            policy_loss.append(-log_prob * R)
        optimizer.zero_grad()
        policy_loss = torch.cat(policy_loss).sum()
        policy_loss.backward()
        optimizer.step()

    return scores

def main():
    n_runs = 10
    belavkin_all_scores = []
    adam_all_scores = []

    for _ in range(n_runs):
        belavkin_scores = train(BelavkinOptimizer, eta=1e-3, gamma=0.01)
        adam_scores = train(torch.optim.Adam, lr=1e-2)
        belavkin_all_scores.append(np.mean(belavkin_scores))
        adam_all_scores.append(np.mean(adam_scores))

    print(f"Belavkin Optimizer - Average score over {n_runs} runs: {np.mean(belavkin_all_scores):.2f} +/- {np.std(belavkin_all_scores):.2f}")
    print(f"Adam Optimizer - Average score over {n_runs} runs: {np.mean(adam_all_scores):.2f} +/- {np.std(adam_all_scores):.2f}")

if __name__ == '__main__':
    main()
