import yaml
import random
import torch
from tqdm import tqdm
import gymnasium as gym

from train_setup.seed_all import seed_all, test_seed
from algorithms.soft_actor_critic import SoftActorCritic
from evaluation.logger import Logger
from train_setup.env_wrapper import EnvWrapper
def rl_loop():
    with open('train_setup/config.yaml', 'r') as file:
        config = yaml.safe_load(file)
    seed_all(config['SEED'])
    env = gym.make(config["ENVIRONMENT"], render_mode=config["RENDER_MODE"],
                   max_episode_steps=config["EPISODE_STOP"])
    env = EnvWrapper(env)
    agent = SoftActorCritic(config, env)
    logger = Logger(env, config)
    for episode in tqdm(range(config["EPISODES"]), desc='Training Process',
                        bar_format=logger.set_tqdm(), colour='white'):
        state = env.reset()
        done = torch.tensor(False).unsqueeze(0).unsqueeze(0)
        episode_reward = 0
        episode_loss = 0
        while not done:
            action, _ = agent.actor.sample_action(state)
            next_state, reward, terminated, truncated, info = env.step(action)
            if episode % 50 == 0:
                env.render()
            if terminated or truncated:
                done = torch.tensor(True).unsqueeze(0).unsqueeze(0)
            agent.memory.add_element(state, action, next_state, reward, done)
            state = next_state
            episode_reward += reward
            agent.update_policy()
        logger.step(episode_reward, 0, 0, config)



if __name__ == "__main__":
    rl_loop()
