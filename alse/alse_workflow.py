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

    def grid(self):
        a, b = torch.meshgrid(
            torch.linspace(0, 1, 10),
            torch.linspace(0, 1, 10),
            indexing="xy",
        )
        c = torch.stack(
            (
                torch.reshape(a, (a.shape[0] * a.shape[1],)),
                torch.reshape(b, (b.shape[0] * b.shape[1],)),
            ),
            dim=1,
        )
        return c.unsqueeze(1)

    def next_test_points(self, num_points):
        normalized_bounds = torch.tensor(
            [[0] * len(self.list_of_models), [1] * len(self.list_of_models)], **tkwargs
        )

        list_of_models_temp = self.list_of_models.copy()
        train_x_temp = self.normalized_x.clone().detach()
        train_y_temp = self.train_y.copy()

        for _ in range(num_points):
            model_list = ModelListGP(*[model for model in list_of_models_temp])
            self.eci = ExpectedCoverageImprovement(
                model=model_list,
                constraints=self.y_constraints,
                punchout_radius=self.punchout_radius,
                bounds=normalized_bounds,
                num_samples=512,
            )
            x_next, _ = optimize_acqf(
                acq_function=self.eci,
                bounds=normalized_bounds,
                q=1,
                num_restarts=10,
                raw_samples=512,
                # fixed_features_list=[{2: 0, 2: 1, 3: 0.9}]
            )
            list_of_models_temp = []
            print(train_x_temp)
            print(x_next)
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

    def get_acq_val_grid(self):
        return self.eci.forward(self.grid())