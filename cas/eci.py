import torch
import gpytorch
from botorch.acquisition.monte_carlo import MCAcquisitionFunction
from botorch.acquisition.objective import IdentityMCObjective
from botorch.utils.sampling import sample_hypersphere
from botorch.utils.transforms import t_batch_mode_transform
from cas.utils import smooth_mask, smooth_box_mask


class ExpectedCoverageImprovement(MCAcquisitionFunction):
    def __init__(
        self,
        model,
        constraints,
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
        assert len(constraints) == model.num_outputs
        assert all(direction in ("gt", "lt") for direction, _ in constraints)
        assert punchout_radius > 0
        self.constraints = constraints
        self.punchout_radius = punchout_radius
        self.bounds = bounds
        self.base_points = self.train_inputs
        self.ball_of_points = self._generate_ball_of_points(
            num_samples=num_samples,
            radius=punchout_radius,
            device=bounds.device,
            dtype=bounds.dtype,
        )
        self._thresholds = torch.tensor(
            [threshold for _, threshold in self.constraints]
        ).to(bounds)
        # assert (
        #     all(ub > lb for lb, ub in self.bounds.T) and len(self.bounds.T) == self.dim
        # )

    @property
    def num_outputs(self):
        return self.model.num_outputs

    @property
    def dim(self):
        return self.train_inputs.shape[-1]

    @property
    def train_inputs(self):
        return self.model.models[0].train_inputs[0]

    def _generate_ball_of_points(
        self, num_samples, radius, device=None, dtype=torch.float
    ):
        """Creates a ball of points to be used for MC."""
        tkwargs = {"device": device, "dtype": dtype}
        z = sample_hypersphere(d=self.dim, n=num_samples, qmc=True, **tkwargs)
        r = torch.rand(num_samples, 1, **tkwargs) ** (1 / self.dim)
        return radius * r * z

    def _get_base_point_mask(self, X):
        distance_matrix = self.model.models[0].covar_module.base_kernel.covar_dist(
            X.float(), self.base_points.float()
        )
        return smooth_mask(distance_matrix, self.punchout_radius)

    def _estimate_probabilities_of_satisfaction_at_points(self, points):
        """Estimate the probability of satisfying the given constraints."""
        final_prob = torch.ones(points.shape[:-1])
        for num in range(len(self.model.models)):
            model = self.model.models[num]
            if model.model_type == "class":
                probabilities = torch.ones(points.shape[:-1])
                for i in range(len(points)):
                    with gpytorch.settings.fast_pred_var(), torch.no_grad():
                        test_dist = model(points[i].float())
                    pred_samples = test_dist.sample(torch.Size((50,))).exp()
                    prob_of_one_point = (
                        pred_samples / pred_samples.sum(-2, keepdim=True)
                    )[:, 1, :].mean(0)
                    probabilities[i] = prob_of_one_point
                final_prob = final_prob * probabilities
            else:
                posterior = model.posterior(X=points)
                mus, sigma2s = posterior.mean, posterior.variance
                dist = torch.distributions.normal.Normal(mus, sigma2s.sqrt())
                norm_cdf = dist.cdf(self._thresholds)
                probs = torch.ones(points.shape[:-1]).to(points)
                direction, _ = self.constraints[num]
                probs = (
                    norm_cdf[..., num] if direction == "lt" else 1 - norm_cdf[..., num]
                )
                final_prob = final_prob * probs
        return final_prob

    @t_batch_mode_transform(expected_q=1)
    def forward(self, X):
        """Evaluate Expected Improvement on the candidate set X."""
        ball_around_X = self.ball_of_points + X
        domain_mask = smooth_mask(
            ball_around_X, self.bounds[0, :], self.bounds[1, :]
        ).prod(dim=-1)
        num_points_in_integral = domain_mask.sum(dim=-1)
        base_point_mask = self._get_base_point_mask(ball_around_X).prod(dim=-1)
        prob = self._estimate_probabilities_of_satisfaction_at_points(ball_around_X)
        masked_prob = prob * domain_mask * base_point_mask
        # print(X)
        # print("prob", torch.mean(prob, axis = 1))
        # print("domain", torch.mean(domain_mask, axis = 1))
        # print("base point", torch.mean(base_point_mask, axis = 1))
        y = masked_prob.sum(dim=-1) / num_points_in_integral
        # print('y', y)
        return y
