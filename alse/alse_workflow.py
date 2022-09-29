import torch, gpytorch
from botorch.models import SingleTaskGP, ModelListGP
from gpytorch.mlls import ExactMarginalLogLikelihood
from gpytorch.constraints import Interval
from botorch.fit import fit_gpytorch_model
from alse.gp_model import fit_gp_model
from alse.eci import ExpectedCoverageImprovement
from alse.utils import normalize, unnormalize
from botorch.optim import optimize_acqf
import copy
from botorch.sampling.samplers import SobolQMCNormalSampler

tkwargs = {
    "device": torch.device("cpu"),
    "dtype": torch.float,
}


class alse:
    def __init__(self, train_x, x_bounds, train_y, y_constraints, punchout_radius=0.1):
        self.train_x = train_x
        self.x_bounds = x_bounds
        self.train_y = train_y
        self.y_constraints = y_constraints
        self.punchout_radius = punchout_radius
        self.grid = None
        self.model_prediction = None

    def initialize_model(self, model_type, **kwargs):
        """Function for initializing the models assuming all models using the same inputs.
        kwargs: *lr* - learning rate
                *num_epoch* - the number of training epoch
        """
        # TODO: should all models use the same lr and num_epoch?
        assert len(model_type) == len(self.train_y)
        self.model_type = model_type
        self.list_of_models = []
        for i in range(len(model_type)):
            self.normalized_x = normalize(self.train_x, self.x_bounds)
            self.list_of_models.append(
                fit_gp_model(
                    model_type[i], self.normalized_x, self.train_y[i], **kwargs
                )
            )
        self.normalized_bounds = torch.tensor(
            [[0] * self.train_x.shape[1], [1] * self.train_x.shape[1]], **tkwargs
        )
        model_list = ModelListGP(*[model for model in self.list_of_models])
        self.eci = ExpectedCoverageImprovement(
            model=model_list,
            constraints=self.y_constraints,
            punchout_radius=self.punchout_radius,
            bounds=self.normalized_bounds,
            num_samples=512,
        )

    def get_grid(self, resolution=20):
        [*X] = torch.meshgrid(
            [
                torch.linspace(0, 1, resolution, **tkwargs)
                for _ in range(self.train_x.shape[1])
            ],
            indexing="xy",
        )
        self.grid = torch.stack(
            [torch.reshape(i, (resolution ** self.train_x.shape[1],)) for i in [*X]],
            dim=1,
        )
        return self.grid

    def next_test_points(self, num_points):

        list_of_models_temp = self.list_of_models.copy()
        train_x_temp = self.normalized_x.clone().detach()
        train_y_temp = self.train_y.copy()

        for _ in range(num_points):
            model_list = ModelListGP(*[model for model in list_of_models_temp])
            self.eci = ExpectedCoverageImprovement(
                model=model_list,
                constraints=self.y_constraints,
                punchout_radius=self.punchout_radius,
                bounds=self.normalized_bounds,
                num_samples=512,
            )
            x_next, _ = optimize_acqf(
                acq_function=self.eci,
                bounds=self.normalized_bounds,
                q=1,
                num_restarts=10,
                raw_samples=512,
                # fixed_features_list=[{2: 0, 2: 1, 3: 0.9}]
            )
            list_of_models_temp = []
            train_x_temp = torch.cat((train_x_temp, x_next))

            for i in range(len(self.model_type)):
                y_on_x_next = model_list.models[i](x_next).loc.unsqueeze(-1)

                train_y_temp[i] = torch.cat((train_y_temp[i], y_on_x_next))
                list_of_models_temp.append(
                    fit_gp_model(self.model_type[i], train_x_temp, train_y_temp[i])
                )
        self.next_batch_test_point = unnormalize(
            train_x_temp[-num_points:], self.x_bounds
        )
        return self.next_batch_test_point

    def get_acq_val_grid(self, resolution=20):
        for i in self.list_of_models:
            i.eval()
        if self.grid == None:
            self.get_grid(resolution)
        return self.eci.forward(self.grid.unsqueeze(1))

    def get_posterior_grid(self, resolution=20):
        """_summary_

        Returns:
            List: a list object with n tensor objects, each tensor is
            of length d, where n is the number of output, and d is the
            number of
        """
        for i in self.list_of_models:
            i.eval()

        if self.grid == None:
            self.get_grid(resolution)
        self.model_prediction = [
            model(self.grid).loc.detach() for model in self.list_of_models
        ]

        in_boundary = [[]] * len(self.model_prediction)
        for i, (direction, value) in enumerate(self.y_constraints):
            if direction == "gt":
                in_boundary[i] = self.model_prediction[i] > value
            else:
                in_boundary[i] = self.model_prediction[i] < value
            if i == 0:
                overlap = in_boundary[0]
            else:
                overlap = overlap & in_boundary[i]
        return self.model_prediction, overlap.float()

    def get_points_mask(self, points_y):
        in_boundary = [[]] * len(self.y_constraints)

        for i, (direction, value) in enumerate(self.y_constraints):
            if direction == "gt":
                in_boundary[i] = points_y[i].flatten() > value
            else:
                in_boundary[i] = points_y[i].flatten() < value
            if i == 0:
                overlap = in_boundary[0]
            else:
                overlap = overlap & in_boundary[i]
        return in_boundary, overlap
