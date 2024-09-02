import gymnasium as gym
from gymnasium import spaces
import torch
import numpy as np
class EnvWrapper(gym.Wrapper):
    def __init__(self, env, config):
        gym.Wrapper.__init__(self, env)
        self.name = config["ENVIRONMENT"]
        if self.name in config["NORMAL_ENV"]:
            env.action_space = spaces.Box(low=-1.0, high=1.0, shape=env.action_space.shape, dtype=np.float32)

    def reset(self, seed: int):
        state = self.env.reset(seed=seed)[0]
        return torch.from_numpy(state).unsqueeze_(dim=0).float()

    def step(self, action):
        action = action.squeeze(0).numpy()
        if self.name == "Pendulum-v1" or self.name == "Pusher-v5":
            action = action * 2.0
        if self.name == "InvertedPendulum-v5":
            action = action * 3.0
        next_state, reward, terminated, truncated, info = self.env.step(action)
        next_state = torch.from_numpy(next_state).unsqueeze_(dim=0).float()
        reward = torch.tensor(reward).view(1, -1)
        terminated = torch.tensor(terminated).view(1, -1)
        truncated = torch.tensor(truncated).view(1, -1)

        return next_state, reward, terminated, truncated, info