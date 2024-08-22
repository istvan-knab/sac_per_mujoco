import numpy as np
import torch
import random
def seed_all(seed: int, env) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    env.action_space.seed(seed)
    env.observation_space.seed(seed)

def test_seed():
    num1 = random.randint(0,100)
    num2 = np.random.rand()
    num3 = torch.rand(3)
    print(num1, num2, num3)