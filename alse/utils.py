import torch
import copy

import pandas


def read_excel(file_path, x_names, y_names):
    # NaN to 0
    consolidated_data = pandas.read_excel(f"{file_path}").fillna(0).loc
    # Input parameters
    input_param = []
    for xname in x_names:
        input_param.append(torch.tensor(consolidated_data[:, f"{xname}"]))

    # Output parameters
    output_param = []
    for yname in y_names:
        output_param.append(
            (torch.tensor((consolidated_data[:, f"{yname}"]))).unsqueeze(-1)
        )

    X = torch.stack(tuple(input_param), -1)
    return X.double(), *output_param


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
    x_copy = x.clone().detach()
    for i in range(x_copy.shape[1]):
        x_copy[:, i] = x_copy[:, i] * (bounds[1][i] - bounds[0][i]) + bounds[0][i]
    return x_copy
