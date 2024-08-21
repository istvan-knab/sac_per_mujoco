import yaml
def rl_loop():
    with open('train_setup/config.yaml', 'r') as file:
        config = yaml.safe_load(file)
if __name__ == "__main__":
    rl_loop()
