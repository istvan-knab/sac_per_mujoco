import neptune

class Logger:
    def __init__(self, environment, config):
        self.run = neptune.init_run(
            project="KovariProductions/per-mujoco-sac",
            api_token="eyJhcGlfYWRkcmVzcyI6Imh0dHBzOi8vYXBwLm5lcHR1bmUuYWkiLCJhcGlfdXJsIjoiaHR0cHM6Ly9hcHAubmVwdHVuZS5haSIsImFwaV9rZXkiOiI5ZGUzMWI0ZC01ZDIyLTQwNWQtODQzOS1mNzQ5NTA3YzdmOGUifQ==",
        )  # your credentials
        self.run["sys/tags"].add([config["ENVIRONMENT"], "SAC", config["TRAIN_MODE"],
                                  str(config["SEED"]), str(config["CP"])])
        self.run["algorithm"] = "SAC"
        self.run["environment"] = config["ENVIRONMENT"]
        self.run["EPISODES"] = config["EPISODES"]
        self.run["EARLY_STOP"] = config["EARLY_STOP"]
        self.run["HIDDEN_LAYERS"] = config["HIDDEN_LAYERS"]
        self.run["LR"] = config["LR"]
        self.run["DISCOUNT_FACTOR"] = config["DISCOUNT_FACTOR"]
        self.run["TRAIN_MODE"] = config["TRAIN_MODE"]
        self.run["BUFFER_SIZE"] = config["BUFFER_SIZE"]
        self.run["BATCH_SIZE"] = config["BATCH_SIZE"]
        self.run["DEVICE"] = config["DEVICE"]
        self.run["TAU"] = config["TAU"]
        self.run["SEED"] = config["SEED"]
        self.run["ENTROPY_START"] = config["ENTROPY_START"]
        self.run["ENTROPY_END"] = config["ENTROPY_END"]
        self.run["CP"] = config["CP"]
        self.run["group"] = str(config["SEED"])
        self.run_id = self.run["sys/id"].fetch()
        self.start_training(config)


    def start_training(self, config):
        print("Starting training------------------")
        print(f'Train Mode: {config["TRAIN_MODE"]}')
        if config["TRAIN_MODE"] == "per":
            print(f"f PER ALPHA : {config['PER_ALPHA']}")
        elif config["TRAIN_MODE"] == "ucb":
            print(f" CP : {config['CP']}")
        print(f'Environment: {config["ENVIRONMENT"]}')
        print(f'Neptune ID : {self.run_id}')
        print(f'Device : {config["DEVICE"]}')
        print(f"Learning Rate: {config['LR']}")
        print(f"Discount Factor: {config['DISCOUNT_FACTOR']}")
        print(f'Buffer size: {config["BUFFER_SIZE"]}')
        print(f'Batch size: {config["BATCH_SIZE"]}')

    def neptune_log(self,reward, c_1_loss,c_2_loss, a_loss, episode_step, temperature):
        self.run["train/reward"].append(reward)
        self.run["train/critic_1_loss"].append(c_1_loss)
        self.run["train/critic_2_loss"].append(c_2_loss)
        self.run["train/actor_loss"].append(a_loss)
        self.run["train/episode_step"].append(episode_step)
        self.run["train/temperature"].append(temperature)


    def step(self, reward, c_1_loss,c_2_loss, a_loss, episode_step, temperature):
        self.neptune_log(reward, c_1_loss,c_2_loss, a_loss, episode_step, temperature)

    def set_tqdm(self):
        WHITE = '\033[97m'
        RESET = '\033[0m'
        tqdm_format = f'{WHITE}{{l_bar}}{{bar}}{{r_bar}}{RESET}'
        return tqdm_format

def console_log(reward):
    print('------------------------')
    print('------------------------')
    print("Episode reward:", reward)
