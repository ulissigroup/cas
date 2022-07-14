import os
import gpytorch
from botorch.models.gpytorch import BatchedMultiOutputGPyTorchModel
from botorch.models.utils import multioutput_to_batch_mode_transform
import matplotlib.pyplot as plt
import torch
from botorch.acquisition.monte_carlo import MCAcquisitionFunction
from botorch.acquisition.objective import IdentityMCObjective
from botorch.fit import fit_gpytorch_model
from botorch.models import ModelListGP, SingleTaskGP
from botorch.models.transforms.outcome import Standardize
from botorch.optim import optimize_acqf
from botorch.utils.sampling import sample_hypersphere
from botorch.utils.transforms import t_batch_mode_transform
from gpytorch.constraints import Interval
from gpytorch.likelihoods import GaussianLikelihood
from gpytorch.mlls import ExactMarginalLogLikelihood
from torch.quasirandom import SobolEngine
from alse.utils import smooth_mask, smooth_box_mask
from matplotlib import pyplot as plt
from mpl_toolkits import mplot3d
import numpy as np
import random
from alse.gp_model import DirichletGPModel
from alse.eci import ExpectedCoverageImprovement
from gpytorch.models import ExactGP
from gpytorch.likelihoods import DirichletClassificationLikelihood
from gpytorch.means import ConstantMean
from gpytorch.kernels import ScaleKernel, RBFKernel
from botorch.models.gpytorch import BatchedMultiOutputGPyTorchModel
import random
import torch
from alse.test_function.fxns import yf
from alse.plot import plot_acq_pos_gif
import os

SMOKE_TEST = os.environ.get("SMOKE_TEST")

tkwargs = {
    "device": torch.device("cpu"),
    "dtype": torch.double,
    # "dtype": torch.float,
}


def get_and_fit_gp(X, Y):
    # Find optimal model hyperparameters
    X = X.float()
    likelihood = DirichletClassificationLikelihood(
        Y[:, 0].long(), learn_additional_noise=True
    )
    model = DirichletGPModel(
        X,
        likelihood.transformed_targets,
        likelihood,
        num_classes=likelihood.num_classes,
    )
    model.train()
    likelihood.train()
    # Use the adam optimizer
    optimizer = torch.optim.Adam(
        model.parameters(), lr=0.1
    )

    mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, model)

    for i in range(50):
        # Zero gradients from previous iteration
        optimizer.zero_grad()
        # Output from model
        output = model(X)
        # Calc loss and backprop gradients
        loss = -mll(output, likelihood.transformed_targets).sum()
        loss.backward()
        optimizer.step()
    return model


lb = -3
ub = 3
bounds = torch.tensor([[lb, lb], [ub, ub]], **tkwargs)
punchout_radius = 0.6
threshold = 160
num_init_points = 5
num_total_points = 20

# --------------------------------------- #
# Setting up training points


X = (lb - ub) * torch.rand(num_init_points, 2) + ub
X = X.float()
Y = yf(X)
# --------------------------------------- #
if not os.path.exists("results"):
    os.mkdir("results")
while len(X) < num_total_points:
    # We don't have to normalize X since the domain is [0, 1]^2. Make sure to
    # appropriately adjust the punchout radius if the domain is normalized.
    print("Checkpoint: build gp model")
    gp_models = get_and_fit_gp(X.float(), Y[:, 0:1])
    print("Checkpoint: ECI")
    eci = ExpectedCoverageImprovement(
        model=gp_models,
        punchout_radius=punchout_radius,
        bounds=bounds,
        num_samples=50,
    )

    plot_acq_pos_gif(
        model=gp_models,
        queried_pts_x=X,
        queried_pts_y=Y,
        num_init_points=num_init_points,
        lb=lb,
        ub=ub,
        tkwargs=tkwargs,
    )

    print("Checkpoint: x_next")
    # Switch to eval mode
    gp_models.eval()

    x_next, _ = optimize_acqf(
            acq_function=eci,
            bounds=bounds,
            q=1,
            num_restarts=10,
            raw_samples=512,
        )
    print(f"Got x_next: {x_next}")
    y_next = yf(x_next)

    X = torch.cat((X, x_next))
    Y = torch.cat((Y, y_next))
    print("Added new point to training data")