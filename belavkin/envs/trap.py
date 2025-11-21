
import numpy as np
import gymnasium as gym
from gymnasium import spaces

class TrapEnv(gym.Env):
    """
    A simple GridWorld with a "Trap".

    Map:
    S . . . . G  (Optimal Path: 5 steps)
    . # # # # .
    . . . . . .  (Safe Path: 7 steps)

    But with a twist:
    - The "Optimal Path" has a high "information cost" or "trap" probability if explored randomly.
    - Actually, let's make it a "Local Optima" vs "Global Optima" problem where exploration is dangerous.

    Simpler Layout:
    State is 1D position: 0 to N.
    Start at 0.
    Action: 0 (Stay/Safe), 1 (Forward/Risky).

    If you go Forward:
      - Small probability of "death" (reset to 0 with penalty) if you don't "know" the path (simulated by random key?).
      - Or simply: A "Cliff" environment where epsilon-greedy falls off.

    Let's use a standard CliffWalking-like env but with sparse rewards.

    Grid: 4x12
    Start: (3, 0)
    Goal: (3, 11)
    The "Cliff" is (3, 1..10). Falling yields -100 and reset.
    Standard PPO often fails or takes long.

    Innovation Gating Idea:
    - When agent falls into cliff repeatedly -> Gradients spike (Surprise).
    - IG-RL should increase entropy ("Panic") to try wild jumps or different policies?
    - Actually, high entropy might make it fall MORE?

    Wait, the prompt says: "Panic... when learning becomes unstable... turning collapse into anti-collapse... force policy to reduce output entropy?"
    Re-reading prompt: "automates 'panic mode'... forcefully regularizing... to REDUCE its output entropy" (Zeno Effect).
    Wait, prompt ALSO says: "Panic 'collapse' when delta >> 0... modulate entropy".

    The user's "Green List" says: "modulate entropy/temperature (panic 'collapse' when delta >> 0)".
    Does panic mean High Entropy (Random) or Low Entropy (Freeze)?
    Usually "Panic" = High Entropy (Flail).
    But "Zeno Effect" = Low Entropy (Freeze).

    User's interpretation: "My learning process is unstable. I should act more randomly (increase entropy) to gather different data."

    Okay, I will implement **Panic = Increase Entropy**.

    Environment: **SparseTrap**.
    - Agent must press a sequence of buttons.
    - Wrong button = Reset.
    - Reward only at end.
    - Gradients will be zero or chaotic.
    """

    def __init__(self, length=10):
        super().__init__()
        self.length = length
        self.observation_space = spaces.Box(low=0, high=1, shape=(length+1,), dtype=np.float32)
        self.action_space = spaces.Discrete(2) # 0: Left, 1: Right
        self.state = 0
        self.max_steps = length * 2
        self.steps = 0

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.state = 0
        self.steps = 0
        return self._get_obs(), {}

    def _get_obs(self):
        obs = np.zeros(self.length + 1, dtype=np.float32)
        obs[self.state] = 1.0
        return obs

    def step(self, action):
        self.steps += 1
        reward = 0
        terminated = False
        truncated = False

        if action == 1: # Right
            self.state += 1
        else: # Left (or stay?)
            # self.state = max(0, self.state - 1)
            pass # No-op or stay

        # Trap: If you are at state floor(length/2) and take action 1, you die unless...
        # Let's just make it a simple Chain environment.
        # State 0 -> 1 -> ... -> N (Reward 1)
        # Action 0 returns to 0 (Reward 0.001 small trap).
        # This is the "N-Chain" problem. Deep RL struggles because "return to start" is a local attractor.

        if action == 0:
            # "Safe" small reward, loop back
            reward = 0.01
            self.state = 0
        else:
            # Risky forward
            reward = 0

        if self.state == self.length:
            reward = 10.0 # Big reward
            terminated = True

        if self.steps >= self.max_steps:
            truncated = True

        return self._get_obs(), reward, terminated, truncated, {}
