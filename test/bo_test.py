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
from matplotlib import pyplot as plt
from mpl_toolkits import mplot3d
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
import random
from gpytorch.models import ExactGP
from gpytorch.likelihoods import DirichletClassificationLikelihood
from gpytorch.means import ConstantMean
from gpytorch.kernels import ScaleKernel, RBFKernel
from alse.eci import ExpectedCoverageImprovement
from alse.gp_model import DirichletGPModel
SMOKE_TEST = os.environ.get("SMOKE_TEST")
# def f(x1, x2): return np.sin(x1+x2)+(x1-x2)**2-1.5*x1+2.5*x2+1
def f(x1, x2): return (x1**2+x2-11)**2+(x1+x2**2-7)**2
x1 = np.linspace(-3, 3)
x2 = np.linspace(-3, 3)
X1, X2 = np.meshgrid(x1, x2)

tkwargs = {
    # "device": torch.device("cuda" if torch.cuda.is_available() else "cpu"),
    "device": torch.device("cpu"),
    "dtype": torch.double,
}





def get_and_fit_gp(X, Y):
    """Simple method for creating a GP with one output dimension.

    X is assumed to be in [0, 1]^d.
    """
#     assert Y.ndim == 2 and Y.shape[-1] == 1
    # likelihood = GaussianLikelihood(noise_constraint=Interval(1e-6, 1e-3))  # Noise-free
    # octf = Standardize(m=Y.shape[-1])
    #-----------------------------------------------------------------------#
    # gp = SingleTaskGP(X, Y, likelihood=likelihood, outcome_transform=octf)
    #-----------------------------------------------------------------------#
    likelihood = DirichletClassificationLikelihood(Y.long(), learn_additional_noise=True)

    gp = DirichletGPModel(X, likelihood.transformed_targets, likelihood, num_classes=likelihood.num_classes)

    mll = ExactMarginalLogLikelihood(gp.likelihood, gp)
    from torch.optim import SGD

    optimizer = SGD([{'params': gp.parameters()}], lr=0.1)
#     fit_gpytorch_model(mll)
    NUM_EPOCHS = 150

    gp.train()
    likelihood.train()
    for epoch in range(NUM_EPOCHS):
        # clear gradients
        optimizer.zero_grad()
        # forward pass through the model to obtain the output MultivariateNormal
        output = gp(X)
        # Compute negative marginal log likelihood
        loss = - mll(output, likelihood.transformed_targets.double()).sum()
        # back prop gradients
        loss.backward()
        # print every 10 iterations
        # if (epoch + 1) % 10 == 0:
        #     print(
        #         f"Epoch {epoch+1:>3}/{NUM_EPOCHS} - Loss: {loss.item():>4.3f} "
        #         f"lengthscale: {gp.covar_module.base_kernel.lengthscale} " 
        #         f"noise: {gp.likelihood.noise}" 
        #      )
        optimizer.step()
    return gp

def yf(x):
    # v = torch.sin(x[:,0]+x[:,1])+(x[:,0]-x[:,1])**2-1.5*x[:,0]+2.5*x[:,1]+1+random.random()
    v = torch.square((torch.square(x[:,0])+x[:,1]-11))+torch.square((x[:,0]+torch.square(x[:,1])-7))
    for i in range(len(v)):
        if v[i] > 100:
            v[i] = 1
        else:
            v[i] = 0
    # return v.reshape(-1, 1).int()
    return v.int()

bounds = torch.tensor([[-3, -3], [3, 3]], **tkwargs)
lb, ub = bounds
dim = len(lb)
punchout_radius = 0.6

num_init_points = 5
num_total_points = 25 
X = lb + (ub - lb) * SobolEngine(dim, scramble=True).draw(num_init_points).to(**tkwargs)
Y = yf(X)
# # Y = yf(X.cpu())
# # plt.scatter(X.numpy()[:, 0], X.numpy()[:, 1], c=Y[:,0])
# plt.scatter(X.cpu().numpy()[:, 0], X.cpu().numpy()[:, 1], c=Y.cpu()[:,0])


# constraints = [("lt", 0), ("gt", -10)]
constraints = [("lt", 0.5)]
# normalization
# mean = X.mean(dim=-2, keepdim=True)
# std = X.std(dim=-2, keepdim=True) + 1e-6 # prevent dividing by 0
# X = (X - mean) / std
# print(X)
# print(Y)
while len(X) < num_total_points:
    # We don't have to normalize X since the domain is [0, 1]^2. Make sure to
    # appropriately adjust the punchout radius if the domain is normalized.
    gp_models = get_and_fit_gp(X, Y)
    # model_list_gp = ModelListGP(gp_models[0])
    # train_X = X.unsqueeze(-3).expand(X.shape[:-2] + torch.Size([2]) + X.shape[-2:])
    eci = ExpectedCoverageImprovement(
        model=gp_models,
        constraints=constraints,
        punchout_radius=punchout_radius,
        bounds=bounds,
        num_samples=4 if not SMOKE_TEST else 4,
    )
    x_next, _ = optimize_acqf(
        acq_function=eci,
        bounds=bounds,
        q=1,
        num_restarts=10 if not SMOKE_TEST else 2,
        raw_samples=4 if not SMOKE_TEST else 4,
    )
    y_next = yf(x_next)
    X = torch.cat((X, x_next))
    Y = torch.cat((Y, y_next))