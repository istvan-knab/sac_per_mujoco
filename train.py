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

def replace_non_compliant_json_values(arr):
    return [0 if x != x or x == float('inf') or x == float('-inf') else x for x in arr]

def rl_loop():
    with open('train_setup/config.yaml', 'r') as file:
        config = yaml.safe_load(file)
    if config['RENDER'] == 'yes':
        env = gym.make(config["ENVIRONMENT"],render_mode = config["RENDER_MODE"])
    else:
        env = gym.make(config["ENVIRONMENT"])
    seed_all(config['SEED'], env)
    env = EnvWrapper(env)
    agent = SoftActorCritic(config, env)
    logger = Logger(env, config)
    last_steps = deque([], maxlen=10)
    for episode in tqdm(range(config["EPISODES"]), desc='Training Process',
                        bar_format=logger.set_tqdm(), colour='white'):
        state = env.reset(config["SEED"])
        done = torch.tensor(False).unsqueeze(0).unsqueeze(0)
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
            if config['RENDER'] == 'yes':
                env.render()
            if terminated or truncated:
                done = torch.tensor(True).unsqueeze(0).unsqueeze(0)
            agent.memory.add_element(state, action, next_state, reward, done)
            state = next_state
            episode_reward += reward
            critic_1_loss, critic_2_loss, actor_loss = agent.update_policy()
            episode_critic_1_loss += critic_1_loss
            episode_critic_2_loss += critic_2_loss
            episode_actor_loss += actor_loss
            losses = [episode_critic_1_loss, episode_critic_2_loss, episode_actor_loss]
            losses = replace_non_compliant_json_values(losses)
        last_steps.append(episode_reward)
        logger.step(episode_reward, losses[0], losses[1], losses[2], config, step)
        break_flag = check_early_stopping(np.array(last_steps), config['EARLY_STOP'])
        if break_flag:
            break

    torch.save(agent.actor, "models" + "/" + str(logger.run_id) + ".pth")


if __name__ == "__main__":
    rl_loop()
