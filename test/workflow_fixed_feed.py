# -*- coding: utf-8 -*-
"""workflow_fixed_feed.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/11S_xLW5xHReXf3Gv9eRCZAgVvkZKrvFz
"""

import matplotlib.pyplot as plt
import torch
from torch.quasirandom import SobolEngine

import numpy as np
from alse.accuracy import get_accuracy
from alse.utils import identify_samples_which_satisfy_constraints, store_param, transform, un_transform
from mpl_toolkits import mplot3d
# from matplotlib import cm
from alse.print_workflow.read_excel_fixed_feed import read_excel
from alse.print_workflow.init_model import initialize_models
from alse.one_iter_eci_fixed_feed import one_iter_eci
from alse.print_workflow.generate_test_points_fixed_feed import generate_test_points

tkwargs = {
    "device": torch.device("cpu"),
    "dtype": torch.float,
}

"""Read data"""

X, width, pow_cap, wth = read_excel()

wth

import ipympl

fig = plt.figure(figsize=(8, 8))
ax = plt.axes(projection='3d')
ax.set_xlabel("Power")
ax.set_ylabel("Velocity")
# ax.set_zlabel("Spot size")
# ax.scatter(X[:,0], X[:,1], X[:,2], c=X[:,3])
img = ax.scatter(X[:,0], X[:,1])
fig.colorbar(img)
plt.show()

# from mpl_toolkits.mplot3d import axes3d

# fig = plt.figure()
# ax = axes3d

# plot = ax.scatter(X[:,0], X[:,1], X[:,2])

# Store for un_transform later
xrange_list, xmin_list = store_param(X, 2)

bounds = torch.tensor([[900, 2700], [600, 1800]], **tkwargs)

X = transform(X, 2, bounds)

# You can verify the Normalization is correct by undoing the above step and comparing to original data
# X = un_transform(X, xrange_list, xmin_list, 4)

X

list_of_models = initialize_models(X, width, pow_cap, wth) # You can put in arbitrary amount of output parameters

punchout_radius = 0.1
bounds = torch.tensor([[0, 0], [1, 1]], **tkwargs) # Because we normalized
# bounds = torch.tensor([[-0.082,-0.122], [1.074,1.01]], **tkwargs)
lb, ub = bounds

constraints = [("gt", 2.5), ("gt", 0.5), ("gt", 6)]
# normalization
# mean = X.mean(dim=-2, keepdim=True)
# std = X.std(dim=-2, keepdim=True) + 1e-6 # prevent dividing by 0
# X = (X - mean) / std

# How many new points do we want
num_pred = 5
Ytemp = [width, pow_cap, wth] #Store temps for test point generation

X

# un_transform(torch.tensor([[1.074,1.01]]), xrange_list, xmin_list, 2)

"""Max power: 2700W
Min power: 900W
Max speed: 1800 mm/min
Min speed: 600 mm/min
"""

xmin_list = torch.tensor([900, 600])

xrange_list = torch.tensor([1800, 1200])

model, X = generate_test_points(X, num_pred, Ytemp, constraints, punchout_radius, bounds)
print(X)
X = un_transform(X, xrange_list, xmin_list, 2)

new_N_points = X[-num_pred:]
for i in range(new_N_points.shape[0], 0, -1):
    new_power = round(X[-i][0].item(), 2)
    new_velocity = round(X[-i][1].item(), 2)
    # new_spotsize = round(X[-i][2].item(), 2)
    # new_feedrate = round(X[-i][3].item(), 2)
    print(f"New X_{new_N_points.shape[0]+1-i}: Power: {new_power}W, Velocity: {new_velocity}mm/min")

X

feasible_inds_width = (
    identify_samples_which_satisfy_constraints(width, constraints[0])
    .prod(dim=-1)
    .to(torch.bool)
)

feasible_inds_pow_cap = (
    identify_samples_which_satisfy_constraints(pow_cap, constraints[1])
    .prod(dim=-1)
    .to(torch.bool)
)

feasible_inds_wth = (
    identify_samples_which_satisfy_constraints(wth, constraints[2])
    .prod(dim=-1)
    .to(torch.bool)
)

X[:5,0]

Xplt = X[:10,:]

plt.scatter(X.float().numpy()[10:, 0], X.float().numpy()[10:, 1])
# plt.plot(Xplt[feasible_inds_width, 0].cpu(), Xplt[feasible_inds_width, 1].cpu(), "og")
# plt.plot(Xplt[feasible_inds_wth, 0].cpu(), Xplt[feasible_inds_wth, 1].cpu(), "og")
# plt.scatter(X.float().numpy()[5:, 0], X.float().numpy()[5:, 1], color='k')
plt.show()

list_of_models[0]

N1, N2 = 150, 150
Xplt, Yplt = torch.meshgrid(
    torch.linspace(0, 1, N1, **tkwargs), torch.linspace(0, 1, N2, **tkwargs),
    indexing='xy',
)
xplt = torch.stack(
    (
        torch.reshape(Xplt, (Xplt.shape[0] * Xplt.shape[1],)),
        torch.reshape(Yplt, (Yplt.shape[0] * Yplt.shape[1],)),
    ),
    dim=1,
)
Xplt, Yplt = torch.meshgrid(
    torch.linspace(900, 2700, N1, **tkwargs), torch.linspace(600, 1800, N2, **tkwargs),
    indexing='xy',
)

xplt = xplt.double()

# Xplt = un_transform(Xplt.clone(), xrange_list, xmin_list, 2).double()

# plt.subplots(3, figsize=(8, 8))
predicted_width = model[0](xplt).loc.detach().reshape(N1,N2)
predicted_pow_cap = model[1](xplt).loc.detach().reshape(N1,N2)
predicted_wth = model[2](xplt).loc.detach().reshape(N1,N2)

est_width = predicted_width > 2.5
est_pow_cap = predicted_pow_cap > 0.5
est_wth = predicted_wth > 6

reference = (est_width & est_pow_cap & est_wth)
plt.figure(1)
fig, ax = plt.subplots(1,3, figsize=(15, 5))
# Xplt = un_transform(Xplt, xrange_list, xmin_list, 2)
ax[0].contourf(Xplt.cpu(), Yplt.cpu(), predicted_width.cpu(),cmap="Blues", alpha=0.6)
ax[0].contour(Xplt.cpu(), Yplt.cpu(), est_width.cpu())
ax[1].contourf(Xplt.cpu(), Yplt.cpu(), predicted_pow_cap.cpu(),cmap="Blues", alpha=0.6)
ax[1].contour(Xplt.cpu(), Yplt.cpu(), est_pow_cap.cpu())
ax[2].contourf(Xplt.cpu(), Yplt.cpu(), predicted_wth.cpu(),cmap="Blues", alpha=0.6)
ax[2].contour(Xplt.cpu(), Yplt.cpu(), est_wth.cpu())
# ax[0].contourf(Xplt.cpu(), Yplt.cpu(), reference.cpu())

# plt.contourf(Xplt.cpu(), Yplt.cpu(), reference.cpu(), cmap="Blues", alpha=0.6)
plt.xlabel('Velocity (mm/min)')
plt.ylabel('Power (W)')
plt.show()

fig, ax = plt.subplots(figsize=(6, 6))
ax.contourf(Xplt.cpu(), Yplt.cpu(), reference.cpu(),cmap="Blues", alpha=0.6)
ax.scatter(X.float().numpy()[:, 0], X.float().numpy()[:, 1])
plt.show()

# fig = plt.figure(figsize=(8, 8))
# ax = plt.axes(projection='3d')
# ax.set_xlabel("Power")
# ax.set_ylabel("Velocity")
# ax.set_zlabel("Spot size")
# # ax.scatter(X[:,0], X[:,1], X[:,2], c=X[:,3])
# img = ax.scatter(X[:-num_pred,0], X[:-num_pred,1], X[:-num_pred,2], c=X[:-num_pred,3])
# img = ax.scatter(X[-num_pred:,0], X[-num_pred:,1], X[-num_pred:,2], c=X[-num_pred:,3], marker = "x")
# fig.colorbar(img)
# plt.show()