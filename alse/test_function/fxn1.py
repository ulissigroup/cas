import torch
def yf(x, threshold = 160):
    v = (x[:, 0] ** 2 + x[:, 1] - 11) ** 2 + (x[:, 0] + x[:, 1] ** 2 - 7) ** 2
    for i in range(len(v)):
        if v[i] > threshold:
            v[i] = 1
        else:
            v[i] = 0
    return torch.stack((v, v), dim=-1)