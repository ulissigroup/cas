from alse.alse import Alse
import math
import torch
import numpy as np
import gpytorch
from matplotlib import pyplot as plt

def gen_data(num_data, seed = 2022):
    torch.random.manual_seed(seed)

    x = torch.randn(num_data,1)
    y = torch.randn(num_data,1)
    u = torch.rand(1)
    data_fn = lambda x, y: 1 * torch.sin(0.15 * u * 3.1415 * (x + y)) + 1
    latent_fn = data_fn(x, y)
    z = torch.round(latent_fn).long().squeeze()
    return torch.cat((x,y),dim=1), z, data_fn

train_x, train_y, genfn = gen_data(100)
d1 = np.linspace(-3, 3, 20)
d2 = np.linspace(-3, 3, 20)
alse_test = Alse(train_x, train_y, d1, d2)
alse_test.predict()