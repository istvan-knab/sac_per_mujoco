import numpy as np
import torch
import random
def seed_all(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True)

def test_seed():
    num1 = random.randint(0,100)
    num2 = np.random.rand()
    num3 = torch.rand(3)
    print(num1, num2, num3)