import torch
from botorch.models import ModelListGP
from botorch.optim import optimize_acqf_mixed
from alse.eci_fixed_feed import ExpectedCoverageImprovement
from alse.print_workflow.init_model import initialize_models

def one_iter_eci(X, constraints, punchout_radius, bounds, *Y):
    list_of_models = initialize_models(X, Y[0], Y[1], Y[2])
    assert len(Y) == len(list_of_models)
    model_list_gp = ModelListGP(*[model for model in list_of_models])

    eci = ExpectedCoverageImprovement(
        model=model_list_gp,
        constraints=constraints,
        punchout_radius=punchout_radius,
        bounds=bounds,
        num_samples=512,
    )
    for i in range(len(Y)):
        list_of_models[i].eval()
    print("fixed features")
    x_next, _ = optimize_acqf_mixed(
        acq_function=eci,
        bounds=bounds,
        q=1,
        num_restarts=10,
        raw_samples=512,
        fixed_features_list=[{3: 0.5}]
    )

    return torch.cat((X, x_next)), list_of_models