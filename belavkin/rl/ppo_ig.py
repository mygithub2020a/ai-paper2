
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.distributions import Categorical
from belavkin.envs.trap import TrapEnv

class PPO_IG(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=64, lr=3e-4,
                 innovation_threshold=0.01, panic_entropy_boost=0.1):
        super().__init__()
        self.critic = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1)
        )
        self.actor = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, action_dim),
            nn.Softmax(dim=-1)
        )

        self.optimizer = optim.Adam(self.parameters(), lr=lr)
        self.innovation_threshold = innovation_threshold
        self.panic_boost = panic_entropy_boost

        # Innovation tracking
        self.grad_ema = None
        self.grad_innov_sq_ema = 0.0
        self.alpha = 0.99

    def get_action(self, state, panic_mode=False):
        state = torch.FloatTensor(state).unsqueeze(0)
        probs = self.actor(state)
        dist = Categorical(probs)

        # If Panic Mode, we artificially flatten the distribution (increase entropy)
        if panic_mode:
            # Mix with uniform
            uniform = torch.ones_like(probs) / probs.shape[-1]
            # Simple mix: 0.5 * probs + 0.5 * uniform
            # Or better: temperature scaling?
            # Let's use the prompt's "Entropy/Temperature" idea.
            # Standard logits -> logits / temp.
            # Since we have softmax output already, we can't easily adjust temp.
            # Let's just sample from a higher entropy distribution.
            # Implementation: epsilon-greedy for "panic"
            if torch.rand(1).item() < self.panic_boost:
                 action = torch.randint(0, probs.shape[-1], (1,))
                 return action.item(), dist.log_prob(action), dist.entropy()

        action = dist.sample()
        return action.item(), dist.log_prob(action), dist.entropy()

    def compute_innovation(self):
        # Compute gradient vector
        grads = []
        for p in self.parameters():
            if p.grad is not None:
                grads.append(p.grad.view(-1))

        if not grads:
            return 0.0

        full_grad = torch.cat(grads)

        if self.grad_ema is None:
            self.grad_ema = torch.zeros_like(full_grad)

        # Innovation: g - m
        innovation = full_grad - self.grad_ema
        innov_sq = innovation.pow(2).mean().item()

        # Update EMA
        self.grad_ema.mul_(self.alpha).add_(full_grad, alpha=1-self.alpha)

        return innov_sq

def train_ppo(env_name="Trap", steps=5000, use_ig=False):
    env = TrapEnv(length=5) # N-Chain 5
    model = PPO_IG(env.observation_space.shape[0], env.action_space.n)

    # Hyperparams
    gamma = 0.99
    eps_clip = 0.2

    total_rewards = []
    innovations = []

    state, _ = env.reset()

    for step in range(steps):
        # Rollout
        rollout = []
        for _ in range(200): # Batch size
            # Determine if Panic
            # Use previous innovation value
            panic = False
            if use_ig and len(innovations) > 0:
                # Normalize innovation relative to history?
                # Simple threshold
                if innovations[-1] > model.innovation_threshold:
                    panic = True

            action, log_prob, entropy = model.get_action(state, panic_mode=panic)
            next_state, reward, done, trunc, _ = env.step(action)

            rollout.append((state, action, reward, log_prob, done))

            state = next_state
            if done or trunc:
                state, _ = env.reset()

        # Update
        states, actions, rewards, old_log_probs, dones = zip(*rollout)

        # Discounted rewards
        returns = []
        discounted_sum = 0
        for r, is_done in zip(reversed(rewards), reversed(dones)):
            if is_done:
                discounted_sum = 0
            discounted_sum = r + gamma * discounted_sum
            returns.insert(0, discounted_sum)

        returns = torch.tensor(returns, dtype=torch.float32)
        returns = (returns - returns.mean()) / (returns.std() + 1e-7)

        states_t = torch.FloatTensor(np.array(states))
        actions_t = torch.LongTensor(actions)
        old_log_probs_t = torch.cat(old_log_probs)

        # Optimization steps (K epochs)
        for _ in range(4):
            # Evaluate
            probs = model.actor(states_t)
            dist = Categorical(probs)
            log_probs = dist.log_prob(actions_t)
            entropy = dist.entropy().mean()

            values = model.critic(states_t).squeeze()
            advantage = returns - values.detach()

            # Ratio
            ratio = torch.exp(log_probs - old_log_probs_t.detach())
            surr1 = ratio * advantage
            surr2 = torch.clamp(ratio, 1-eps_clip, 1+eps_clip) * advantage

            loss = -torch.min(surr1, surr2).mean() + 0.5 * nn.MSELoss()(values, returns) - 0.01 * entropy

            model.optimizer.zero_grad()
            loss.backward()

            # Compute Innovation BEFORE Step (on gradients)
            innov = model.compute_innovation()

            model.optimizer.step()

        innovations.append(innov)
        avg_rew = sum(rewards)
        total_rewards.append(avg_rew)

    return total_rewards, innovations

if __name__ == "__main__":
    # Simple test run
    print("Training Baseline PPO...")
    rew_base, _ = train_ppo(use_ig=False, steps=50) # Short run
    print(f"Baseline Mean Reward: {np.mean(rew_base[-10:])}")

    print("Training IG-PPO...")
    rew_ig, innovs = train_ppo(use_ig=True, steps=50)
    print(f"IG-PPO Mean Reward: {np.mean(rew_ig[-10:])}")
