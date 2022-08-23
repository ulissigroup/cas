import matplotlib.pyplot as plt
import torch
from torch.quasirandom import SobolEngine

import numpy as np
from alse.accuracy import get_accuracy
from alse.utils import identify_samples_which_satisfy_constraints, normalize, unnormalize
from mpl_toolkits import mplot3d
# from matplotlib import cm
from alse.utils import read_excel
from alse.alse import alse
import copy
tkwargs = {
    "device": torch.device("cpu"),
    "dtype": torch.float,
}

X, width, pow_cap, wth = read_excel("/home/jovyan/alse/test_data/8_4_data.xlsx",["P (W)", "V (mm/min)"], ["width (mm)", "powder_cap", "wth"])
bounds = torch.tensor([[900, 600], [2700, 1800]])
constraints = [("gt", 2.5), ("gt", 0.65), ("gt", 6)]

algo = alse(X, bounds, [width, pow_cap, wth], constraints)

algo.initialize_model(["reg", "reg", "reg"])

print(algo.next_test_points(5))