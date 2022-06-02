import os
import gpytorch
from botorch.models.gpytorch import BatchedMultiOutputGPyTorchModel
from botorch.models.utils import multioutput_to_batch_mode_transform
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
class DirichletGPModel(ExactGP, BatchedMultiOutputGPyTorchModel):
    def __init__(self, train_x, train_y, likelihood, num_classes):
        # super(DirichletGPModel, self).__init__(train_x, train_y, likelihood)

        #-----------------------------------------------------------------------#
#         train_X, train_Y, _ = multioutput_to_batch_mode_transform(
#             train_X=train_x, train_Y=train_y, num_outputs=num_classes
#         )
#         self.train_X = train_X
#         self.train_X_raw = train_x
#         self.train_Y = train_Y
#         self.train_Y_raw = train_y
        self._set_dimensions(train_X=train_x, train_Y=train_y)
        super().__init__(train_x, train_y, likelihood)

        # self._set_dimensions(train_X=train_x, train_Y=train_y)
        # super().__init__(train_x, train_y, likelihood)

        self.mean_module = ConstantMean(batch_shape=torch.Size([num_classes,]))
        self.covar_module = ScaleKernel(
            RBFKernel(batch_shape=torch.Size([num_classes,])),
            batch_shape=torch.Size([num_classes,]),
        )
        #-----------------------------------------------------------------------#


    def forward(self, x):
        # forward_X = x.unsqueeze(-3).expand(
        #     x.shape[:-2] + torch.Size([self.num_outputs]) + x.shape[-2:]
        # )
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
#         print(mean_x.shape[:-1])
#         print(covar_x.shape[:-2])
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)