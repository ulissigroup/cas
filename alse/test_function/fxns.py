import torch


def yf(x, threshold = 160):
    v = (((x[:,0]**2+x[:,1]-11)**2+(x[:,0]+x[:,1]**2-7)**2)>threshold).float()
    return torch.stack((v, v), dim=-1)

def hills(x):
    v = 0.5*torch.cos(x[:,0]*2-torch.pi)+torch.sin(x[:,1]+torch.pi/2)
    
    return torch.stack((v, v), dim=-1)

def circle(x):
    v = -(x[:,0] ** 2 + x[:,1] ** 2) + 25
    return torch.stack((v, v), dim=-1)