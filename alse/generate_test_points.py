import torch
from alse.one_iter_eci import one_iter_eci

def generate_test_points(X, num_pred, Ytemp, constraints, punchout_radius, bounds):
    for i in range(num_pred):
        for i in range(len(Ytemp)):     # Detach each element
            Ytemp[i] = Ytemp[i].detach()

        X, models = one_iter_eci(X, constraints, punchout_radius, bounds, *Ytemp)

        for i in range(len(Ytemp)):
            Ytemp[i] = torch.cat((Ytemp[i], models[i](X[-1].unsqueeze(0)).loc.unsqueeze(-1)))

        # We might need this if we encounter classification model
        # _compare = (list_of_models[2](X[-1].unsqueeze(0).float()).loc[0] > list_of_models[2](X[-1].unsqueeze(0).float()).loc[1]).long().unsqueeze(0)

    return X
