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
from alse.utils import smooth_mask, smooth_box_mask
class ExpectedCoverageImprovement(MCAcquisitionFunction):
    def __init__(
        self,
        model,
        punchout_radius,
        bounds,
        num_samples=512,
        **kwargs,
    ):
        """Expected Coverage Improvement (q=1 required, analytic)

        Right now, we assume that all the models in the ModelListGP have
        the same training inputs.

        Args:
            model: A ModelListGP object containing models matching the corresponding constraints.
                All models are assumed to have the same training data.
            constraints: List containing 2-tuples with (direction, value), e.g.,
                [('gt', 3), ('lt', 4)]. It is necessary that
                len(constraints) == model.num_outputs.
            punchout_radius: Positive value defining the desired minimum distance between points
            bounds: torch.tensor whose first row is the lower bounds and second row is the upper bounds
            num_samples: Number of samples for MC integration
        """
        super().__init__(model=model, objective=IdentityMCObjective(), **kwargs)
        assert punchout_radius > 0
        self.punchout_radius = punchout_radius
        self.bounds = bounds
        self.base_points = self.train_inputs
        self.ball_of_points = self._generate_ball_of_points(
            num_samples=num_samples,
            radius=punchout_radius,
            device=bounds.device,
            dtype=bounds.dtype,
        )
        assert (
            all(ub > lb for lb, ub in self.bounds.T) and len(self.bounds.T) == self.dim
        )

    @property
    def num_outputs(self):
        return self.model.num_outputs

    @property
    def dim(self):
        return self.train_inputs.shape[-1]

    @property
    def train_inputs(self):
        return self.model.train_inputs[0]

    def _generate_ball_of_points(
        self, num_samples, radius, device=None, dtype=torch.double
    ):
        """Creates a ball of points to be used for MC."""
        tkwargs = {"device": device, "dtype": dtype}
        z = sample_hypersphere(d=self.dim, n=num_samples, qmc=True, **tkwargs) # Not using self.dim
        r = torch.rand(num_samples, 1, **tkwargs) ** (1 / self.dim)
        
        return radius * r * z

    def _get_base_point_mask(self, X):
        distance_matrix = self.model.covar_module.base_kernel.covar_dist(
            X, self.base_points.double()
        )   # Note to self: self.base_points is fp32
            # Should standardize all to fp64?
        return smooth_mask(distance_matrix, self.punchout_radius)

    def _estimate_probabilities_of_satisfaction_at_points(self, points):
            print("Estimating probabilities")
            probabilities = torch.zeros((points.shape[0:2]))
            for i in range(len(points)):
                with gpytorch.settings.fast_pred_var(), torch.no_grad():
                    test_dist = self.model(points[i].float())
                pred_samples = test_dist.sample(torch.Size((50,))).exp()
                prob_of_one_point = (pred_samples / pred_samples.sum(-2, keepdim=True))[:,1,:].mean(0)
                probabilities[i] = prob_of_one_point
            return probabilities

    @t_batch_mode_transform(expected_q=1)
    def forward(self, X):
        """Evaluate Expected Improvement on the candidate set X."""
        ball_around_X = self.ball_of_points + X
        domain_mask = smooth_mask(
            ball_around_X, self.bounds[0, :], self.bounds[1, :]
        ).prod(dim=-1)
        # print(f"domain mask: {domain_mask.shape}")
        num_points_in_integral = domain_mask.sum(dim=-1)
        # print(f"num_points_in_integral: {num_points_in_integral.shape}")
        # print("Right before base_point_mask")
        # print(f"ball dtype: {ball_around_X.dtype}")
        # print(f"ball shape: {ball_around_X.shape}")
        base_point_mask = self._get_base_point_mask(ball_around_X).prod(dim=-1)
        # print(f"base_point_mask: {base_point_mask.shape}")
        prob = self._estimate_probabilities_of_satisfaction_at_points(ball_around_X)
        # print(f"prob: {prob.shape}")
        masked_prob = prob * domain_mask * base_point_mask
        y = masked_prob.sum(dim=-1) / num_points_in_integral
        return y