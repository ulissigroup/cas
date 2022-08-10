import torch
from alse.one_iter_eci_fixed_feed import one_iter_eci

def generate_test_points(X, num_pred, Ytemp, constraints, punchout_radius, bounds):
    for i in range(num_pred):
        for i in range(len(Ytemp)):     # Detach each element
            Ytemp[i] = Ytemp[i].detach()

        X, list_of_models = one_iter_eci(X, constraints, punchout_radius, bounds, *Ytemp)

        for i in range(len(Ytemp)):
            Ytemp[i] = torch.cat((Ytemp[i], list_of_models[i](X[-1].unsqueeze(0)).loc.unsqueeze(-1)))
            
        # This part is hardcoded
        # Ytemp[0] = torch.cat((Ytemp[0], list_of_models[0](X[-1].unsqueeze(0)).loc.unsqueeze(-1)))
        # Ytemp[1] = torch.cat((Ytemp[1], list_of_models[1](X[-1].unsqueeze(0)).loc.unsqueeze(-1)))

        # This is a mess, I'll clean it up later
        # _compare = (list_of_models[2](X[-1].unsqueeze(0).float()).loc[0] > list_of_models[2](X[-1].unsqueeze(0).float()).loc[1]).long().unsqueeze(0)
        # Ytemp[2] = torch.cat((Ytemp[2], list_of_models[2](X[-1].unsqueeze(0)).loc.unsqueeze(-1)))

    return X
