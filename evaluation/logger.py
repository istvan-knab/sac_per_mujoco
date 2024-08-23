import neptune

class Logger:
    def __init__(self, environment, config):
        self.run = neptune.init_run(
            project="KovariProductions/per-mujoco-sac",
            api_token="eyJhcGlfYWRkcmVzcyI6Imh0dHBzOi8vYXBwLm5lcHR1bmUuYWkiLCJhcGlfdXJsIjoiaHR0cHM6Ly9hcHAubmVwdHVuZS5haSIsImFwaV9rZXkiOiI5ZGUzMWI0ZC01ZDIyLTQwNWQtODQzOS1mNzQ5NTA3YzdmOGUifQ==",
        )  # your credentials

        params = {"learning_rate": config["LR"], "discount_factor": config["DISCOUNT_FACTOR"],
                  "Environment": environment, "batch_size": config["BATCH_SIZE"],
                  "EPISODES": config["EPISODES"],"LearningRate": config["LR"], "GAMMA": config["DISCOUNT_FACTOR"],
                  "Mode": config["TRAIN_MODE"]}
        self.run["algorithm"] = "SAC"
        self.run["environment"] = config["ENVIRONMENT"]
        self.run["EPISODES"] = config["EPISODES"]
        self.run["LR"] = config["LR"]
        self.run["DISCOUNT_FACTOR"] = config["DISCOUNT_FACTOR"]
        self.run["TRAIN_MODE"] = config["TRAIN_MODE"]
        self.run["BUFFER_SIZE"] = config["BUFFER_SIZE"]
        self.run["BATCH_SIZE"] = config["BATCH_SIZE"]
        self.run["DEVICE"] = config["DEVICE"]
        self.run["PER_ALPHA"] = config["PER_ALPHA"]
        self.run["BETA"] = config["BETA"]
        self.run["train/ENTROPY_COEFFICIENT"] = config["ENTROPY_COEFFICIENT"]
        self.run_id = self.run["sys/id"].fetch()
        self.start_training(config)


    def start_training(self, config):
        print("Starting training------------------")
        print(f'Train Mode: {config["TRAIN_MODE"]}')
        print(f'Environment: {config["ENVIRONMENT"]}')
        print(f'Neptune ID : {self.run_id}')
        print(f'Device : {config["DEVICE"]}')
        print(f"Learning Rate: {config['LR']}")
        print(f"Discount Factor: {config['DISCOUNT_FACTOR']}")
        print(f'Buffer size: {config["BUFFER_SIZE"]}')
        print(f'Batch size: {config["BATCH_SIZE"]}')

    def neptune_log(self,reward, c_1_loss,c_2_loss, a_loss, episode_step):
        self.run["train/reward"].append(reward)
        self.run["train/critic_1_loss"].append(c_1_loss)
        self.run["train/critic_2_loss"].append(c_2_loss)
        self.run["train/actor_loss"].append(a_loss)
        self.run["train/episode_step"].append(episode_step)


    def step(self, reward, c_1_loss,c_2_loss, a_loss, config, episode_step):

        self.neptune_log(reward, c_1_loss,c_2_loss, a_loss, episode_step)
        self.run["train/algorithm"] = "SAC"
        self.run["train/environment"] = config["ENVIRONMENT"]
        self.run["train/EPISODES"] = config["EPISODES"]
        self.run["train/LR"] = config["LR"]
        self.run["train/DISCOUNT_FACTOR"] = config["DISCOUNT_FACTOR"]
        self.run["train/TRAIN_MODE"] = config["TRAIN_MODE"]
        self.run["train/BUFFER_SIZE"] = config["BUFFER_SIZE"]
        self.run["train/BATCH_SIZE"] = config["BATCH_SIZE"]
        self.run["train/DEVICE"] = config["DEVICE"]
        self.run["train/PER_ALPHA"] = config["PER_ALPHA"]
        self.run["train/BETA"] = config["BETA"]
        self.run["train/ENTROPY_COEFFICIENT"] = config["ENTROPY_COEFFICIENT"]
    def set_tqdm(self):
        WHITE = '\033[97m'
        RESET = '\033[0m'
        tqdm_format = f'{WHITE}{{l_bar}}{{bar}}{{r_bar}}{RESET}'
        return tqdm_format

def console_log(reward):
    print('------------------------')
    print('------------------------')
    print("Episode reward:", reward)
