import torch
import copy
import numpy as np
import pandas


def read_excel(file_path, x_names, y_names, y_round=None, sheet_name=0):
    if y_round is None:
        y_round = [8] * len(y_names)
    # NaN to 0
    consolidated_data = pandas.read_excel(f"{file_path}", sheet_name).fillna(0).loc
    # Input parameters
    input_param = []
    for xname in x_names:
        input_param.append(torch.tensor(consolidated_data[:, f"{xname}"]))

    # Output parameters
    output_param = []
    for i, yname in enumerate(y_names):
        y_val = torch.tensor((consolidated_data[:, f"{yname}"])).unsqueeze(-1)
        output_param.append(torch.round(y_val, decimals=y_round[i]))
    X = torch.stack(tuple(input_param), -1)
    return X.float(), output_param


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


def get_random_points(bounds, dim, seed=42):
    """Generate random points within the given bounds

    Args:
        bounds (torch.tensor):
        dim (int or tuple): dimension of points to generate

    Returns:
        _type_: _description_
    """
    torch.manual_seed(seed)
    if type(dim) is int:
        return unnormalize(torch.rand(dim, bounds.shape[1]), bounds)
    elif type(dim) is tuple:
        return [
            unnormalize(torch.rand(dim[0], bounds.shape[1]), bounds)
            for _ in range(dim[1])
        ]
    else:
        raise Exception("dim is either an int or a tuple")


# Transform each parameter individually to [0, 1]
def normalize(x, bounds):
    """
    Args:
        x (torch.tensor): input to be normalized. The shape of
        the tensoris n x d, where n is the number of points,
        and d is the number of dimensions
        bounds (torch.tensor): first row is the lower bounds
        and second row is the upper bounds

    Returns:
        torch.tensor: normalized x
    """
    x_copy = x.clone().detach()
    for i in range(x_copy.shape[1]):
        # Linear scaling for each parameter
        x_copy[:, i] = (x_copy[:, i] - bounds[0][i]) / (bounds[1][i] - bounds[0][i])
    return x_copy


# Undo the transform step
def unnormalize(x, bounds):
    """
    Args:
        x (torch.tensor): input to be unnormalized. The shape of
        the tensoris n x d, where n is the number of points,
        and d is the number of dimensions
        bounds (torch.tensor): first row is the lower bounds
        and second row is the upper bounds

    Returns:
        torch.tensor: unnormalized x
    """
    x_copy = x.clone().detach()
    for i in range(x_copy.shape[1]):
        x_copy[:, i] = x_copy[:, i] * (bounds[1][i] - bounds[0][i]) + bounds[0][i]
    return x_copy


def get_hatch_xa(width, height):
    """Calculate the hatch spacing and cross sectional area given the width and height

    Args:
        width (np.array): single-track width
        height (np.array): single-track height

    Returns:
        hatch: hatch spacing
        est_area: estimated cross sectional area
    """
    term_1 = np.square((np.square(width / 2) + height * height) / (2 * height))
    term_2 = np.arcsin(width * height / (np.square(width / 2) + height * height))
    term_3 = -1 * (width / 2) * (np.square(width / 2) - height * height) / (2 * height)
    hatch = (term_1 * term_2 + term_3) / height

    overlap_frac = (width - hatch) / width
    est_area = hatch * height

    return hatch, est_area
