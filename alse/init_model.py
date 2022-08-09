import torch
import pandas
from alse.fit_model import fit_gp_class, fit_gp_reg

def initialize_models(X, *Y):
    num_output_param = len(Y)
    list_of_models = []
    for i in range(num_output_param):
        # if  2 unique values use classification, else regression
        if torch.unique(Y[i]).shape[0] == 2:
            list_of_models.append(fit_gp_class(X, Y[i], lr=0.05, iter=300))
        else: list_of_models.append(fit_gp_reg(X, Y[i]))

    return list_of_models