import yaml
import random
import torch

from train_setup.seed_all import seed_all
def rl_loop():
    with open('train_setup/config.yaml', 'r') as file:
        config = yaml.safe_load(file)
    seed_all(config['SEED'])



if __name__ == "__main__":
    rl_loop()
