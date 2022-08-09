import torch
from botorch.models import ModelListGP
from botorch.optim import optimize_acqf
from alse.eci import ExpectedCoverageImprovement
from alse.init_model import initialize_models

def one_iter_eci(X, constraints, punchout_radius, bounds, *Y):
    models = initialize_models(X, *Y)
    assert len(Y) == len(models)
    model_list_gp = ModelListGP(*[model for model in models])
 
    eci = ExpectedCoverageImprovement(
        model=model_list_gp,
        constraints=constraints,
        punchout_radius=punchout_radius,
        bounds=bounds,
        num_samples=512,
    )
    for i in range(len(Y)):
        models[i].eval()

    x_next, acq_val = optimize_acqf(
        acq_function=eci,
        bounds=bounds,
        q=1,
        num_restarts=10,
        raw_samples=512,
        # fixed_features_list=[{2: 0, 2: 1, 3: 0.9}]
    )
    print(f"x_next: {x_next}")
    print(f"acq_val: {acq_val}")

    return torch.cat((X, x_next)), models