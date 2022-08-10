import torch
import copy

import pandas

def read_excel(file_path):

    consolidated_data = pandas.read_excel(f'{file_path}').loc
    #Input parameters
    power = torch.tensor(consolidated_data[:,"P (W)"])
    velocity = torch.tensor(consolidated_data[:,"V (mm/min)"])
    
    #Output parameters
    pow_cap = (torch.tensor(consolidated_data[:, "powder_cap"])/100).unsqueeze(-1)
    width = (torch.tensor(consolidated_data[:, "width (mm)"])).unsqueeze(-1)
    height = (torch.tensor(consolidated_data[:, "height (mm)"])).unsqueeze(-1)
    wth = width/height
    wth = torch.nan_to_num(wth, nan = torch.rand(1).item())

    X = torch.stack((power, velocity), -1)
    return X, width, pow_cap, wth
    
def smooth_mask(x, a, eps=2e-3):
    """Returns 0ish for x < a and 1ish for x > a"""
    return torch.nn.Sigmoid()((x - a) / eps)


def smooth_box_mask(x, a, b, eps=2e-3):
    """Returns 1ish for a < x < b and 0ish otherwise"""
    return smooth_mask(x, a, eps) - smooth_mask(x, b, eps)


def identify_samples_which_satisfy_constraints(X, constraints):
    """
    Takes in values (a1, ..., ak, o) and returns (a1, ..., ak, o)
    True/False values, where o is the number of outputs.
    """
    successful = torch.ones(X.shape).to(X)

    these_X = X[
        ...,
    ]
    direction, value = constraints
    successful[...,] = (
        these_X < value if direction == "lt" else these_X > value
    )
    return successful


# Transform each parameter individually to [0, 1]
def normalize(x, bounds):
    """
    x: torch.tensor whose shape is n x d, where n is the number of points,
        and d is the number of dimensions
    bounds: torch.tensor whose first row is the lower bounds
        and second row is the upper bounds"""
    x_copy = x.clone().detach()
    for i in range(x_copy.shape[1]):
        # Linear scaling for each parameter
        x_copy[:, i] = (x_copy[:, i] - bounds[0][i]) / (bounds[1][i] - bounds[0][i])
    return x_copy


# Undo the transform step
def unnormalize(x, bounds):
    """x: torch.tensor whose shape is n x d, where n is the number of points,
        and d is the number of dimensions
    bounds: torch.tensor whose first row is the lower bounds
        and second row is the upper bounds"""
    for i in range(x.shape[1]):
        x[:, i] = x[:, i] * (bounds[1][i] - bounds[0][i]) + bounds[0][i]
    return x
