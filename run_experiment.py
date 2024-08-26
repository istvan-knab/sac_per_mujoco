import time
import yaml
import gymnasium as gym
import torch

from train_setup.env_wrapper import EnvWrapper
from evaluation.logger import console_log
def run_test():
    with open('test_setup/config.yaml', 'r') as file:
        config = yaml.safe_load(file)
    env = gym.make(config["ENVIRONMENT"], render_mode="human")
    env = EnvWrapper(env)
    state = env.reset(config["SEED"])
    done = torch.tensor(False).unsqueeze(0).unsqueeze(0)
    agent = torch.load("models/" + config["model"])
    step = 0
    while not done:
        step += 1
        action, _ = agent.sample_action(state)
        next_state, reward, terminated, truncated, info = env.step(action)
        if terminated or truncated:
            done = torch.tensor(True).unsqueeze(0).unsqueeze(0)
        state = next_state
        #time.sleep(0.08)
        console_log(f"Step: {step}, Reward: {0}")

if __name__ == "__main__":
    run_test()