import torch


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


# Store necessary parameters for un_transform
def store_param(X, num_param):
    xrange_list = torch.zeros(num_param)
    xmin_list = torch.zeros(num_param)
    for i in range(num_param):
        xrange_list[i] = X[:, i].max() - X[:, i].min()
        xmin_list[i] = X[:, i].min()
    return xrange_list, xmin_list


# Transform each parameter individually to [0, 1]
def normalize(x, bounds):
    """
    x: torch.tensor whose shape is n x d, where n is the number of points,
        and d is the number of dimensions
    bounds: torch.tensor whose first row is the lower bounds
        and second row is the upper bounds"""
    for i in range(x.shape[1]):
        # Linear scaling for each parameter
        x[:, i] = (x[:, i] - bounds[0][i]) / (bounds[1][i] - bounds[0][i])
    return x


# Undo the transform step
def unnormalize(x, bounds):
    """x: torch.tensor whose shape is n x d, where n is the number of points,
        and d is the number of dimensions
    bounds: torch.tensor whose first row is the lower bounds
        and second row is the upper bounds"""
    for i in range(x.shape[1]):
        x[:, i] = x[:, i] * (bounds[1][i] - bounds[0][i]) + bounds[0][i]
    return x
