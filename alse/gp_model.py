import torch
import gpytorch
from gpytorch.models import ExactGP
from gpytorch.means import ConstantMean
from gpytorch.kernels import ScaleKernel, RBFKernel
from botorch.models import SingleTaskGP
from gpytorch.likelihoods import DirichletClassificationLikelihood, GaussianLikelihood
from gpytorch.constraints import Interval
from gpytorch.mlls import ExactMarginalLogLikelihood



class ClassificationModel(ExactGP):
    def __init__(self, train_x, train_y, likelihood, num_classes):
        super(ClassificationModel, self).__init__(train_x, train_y, likelihood)
        self.mean_module = ConstantMean(batch_shape=torch.Size((num_classes,)))
        self.covar_module = ScaleKernel(
            RBFKernel(batch_shape=torch.Size((num_classes,))),
            batch_shape=torch.Size((num_classes,)),
        )
        self.num_outputs = num_classes
        self.model_type = "class"

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)

class RegressionModel(SingleTaskGP):
    def __init__(self,train_x, train_y, likelihood):
        super().__init__(train_x, train_y, likelihood)
        self.model_type = "reg"


def fit_gp_model(model_type, X, Y, **kwargs):
    try:
        assert X.dtype == torch.float
    except:
        X = X.float()
    if model_type == "class":
        likelihood = DirichletClassificationLikelihood(
            Y[:, 0].long(), learn_additional_noise=True
        )
        model = ClassificationModel(
            X, likelihood.transformed_targets, likelihood, num_classes=2
        )
    elif model_type == "reg":
        likelihood = GaussianLikelihood(
            noise_constraint=Interval(1e-9, 1e-6)
        ) # Noise-free
        model = RegressionModel(X, Y, likelihood=likelihood)
    else:
        raise TypeError(
            "Model type not found, \
        please specify model type as class or reg"
        )

    model.train()
    likelihood.train()
    # TODO: allow customized optimizer
    optimizer = torch.optim.Adam(
        model.parameters(), lr=float(kwargs.get('lr',"0.1"))
    )
    mll = ExactMarginalLogLikelihood(likelihood, model)

    for _ in range(int(kwargs.get('num_epoch',"100"))):
        optimizer.zero_grad()
        output = model(X)
        # TODO: move this to model initialization step?
        if model_type == "class":
            loss = -mll(output, likelihood.transformed_targets).sum()
        else:
            loss = - mll(output, model.train_targets)
        loss.backward()
        optimizer.step()
    return model