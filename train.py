import numpy as np
import yaml
import torch
from tqdm import tqdm
import gymnasium as gym
from collections import deque

from train_setup.seed_all import seed_all, test_seed
from algorithms.soft_actor_critic import SoftActorCritic
from evaluation.logger import Logger
from train_setup.env_wrapper import EnvWrapper

def check_early_stopping(last_steps, stop):
    if last_steps.mean() >= stop :
        return True
    else:
        return False

def select_device(config):
    if torch.backends.mps.is_available():
        return torch.device("mps")
    elif torch.cuda.is_available():
        return torch.device("cuda")
    else:
        return torch.device("cpu")

def rl_loop():
    with open('train_setup/config.yaml', 'r') as file:
        config = yaml.safe_load(file)
    if config['RENDER'] == 'yes':
        env = gym.make(config["ENVIRONMENT"],render_mode = config["RENDER_MODE"])
    else:
        env = gym.make(config["ENVIRONMENT"])
    config["DEVICE"] = select_device(config)
    seed_all(config['SEED'], env)
    config["INPUT_FEATURES"] = env.observation_space.shape[0]
    config["ACTION_SPACE"] = env.action_space.shape[0]
    env = EnvWrapper(env, config)
    agent = SoftActorCritic(config, env)
    logger = Logger(env, config)
    last_steps = deque([], maxlen=60)
    for episode in tqdm(range(config["EPISODES"]), desc='Training Process',
                        bar_format=logger.set_tqdm(), colour='white'):
        state = env.reset(config["SEED"])
        done = np.array([False])
        episode_reward = 0
        episode_critic_1_loss = 0
        episode_critic_2_loss = 0
        episode_actor_loss = 0
        step = 0
        losses = []
        while not done:
            step += 1
            action, _ = agent.actor.sample_action(state)
            next_state, reward, terminated, truncated, info = env.step(action)
            if terminated or truncated:
                done = np.array([True])
            agent.memory.add_element(state, action, next_state, reward, done)
            state = next_state
            episode_reward += reward
            critic_1_loss, critic_2_loss, actor_loss = agent.update_policy()
            episode_critic_1_loss += critic_1_loss
            episode_critic_2_loss += critic_2_loss
            episode_actor_loss += actor_loss
            losses = [episode_critic_1_loss, episode_critic_2_loss, episode_actor_loss]
        last_steps.append(episode_reward)
        logger.step(episode_reward, losses[0], losses[1], losses[2], step, agent.temperature)
        break_flag = check_early_stopping(np.array(last_steps), config['EARLY_STOP'])
        agent.set_entropy()
        if break_flag:
            break

    torch.save(agent.actor, "models" + "/" + str(logger.run_id) + ".pth")


if __name__ == "__main__":
    rl_loop()
