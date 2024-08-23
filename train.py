import yaml
import random
import torch
from tqdm import tqdm
import gymnasium as gym

from train_setup.seed_all import seed_all, test_seed
from algorithms.soft_actor_critic import SoftActorCritic
from evaluation.logger import Logger
from train_setup.env_wrapper import EnvWrapper

def check_early_stopping():
    pass
def rl_loop():
    with open('train_setup/config.yaml', 'r') as file:
        config = yaml.safe_load(file)
    if config['RENDER'] == 'yes':
        env = gym.make(config["ENVIRONMENT"],render_mode = config["RENDER_MODE"],
                       max_episode_steps= config["EPISODE_STOP"])
    else:
        env = gym.make(config["ENVIRONMENT"], max_episode_steps=config["EPISODE_STOP"])
    seed_all(config['SEED'], env)
    env = EnvWrapper(env)
    agent = SoftActorCritic(config, env)
    logger = Logger(env, config)
    for episode in tqdm(range(config["EPISODES"]), desc='Training Process',
                        bar_format=logger.set_tqdm(), colour='white'):
        state = env.reset()
        done = torch.tensor(False).unsqueeze(0).unsqueeze(0)
        episode_reward = 0
        episode_critic_1_loss = 0
        episode_critic_2_loss = 0
        episode_actor_loss = 0
        step = 0
        while not done:
            step += 1
            action, _ = agent.actor.sample_action(state)
            next_state, reward, terminated, truncated, info = env.step(action)
            if config['RENDER'] == 'yes':
                env.render()
            if step == config["EPISODE_STOP"]:
                terminated = True
            if terminated or truncated:
                done = torch.tensor(True).unsqueeze(0).unsqueeze(0)
            agent.memory.add_element(state, action, next_state, reward, done)
            state = next_state
            episode_reward += reward
            critic_1_loss, critic_2_loss, actor_loss = agent.update_policy()
            episode_critic_1_loss += critic_1_loss
            episode_critic_2_loss += critic_2_loss
            episode_actor_loss += actor_loss
        logger.step(episode_reward, episode_critic_1_loss, episode_critic_2_loss,episode_actor_loss, config)

    torch.save(agent.actor, "models" + "/" + str(logger.run_id) + ".pth")


if __name__ == "__main__":
    rl_loop()
