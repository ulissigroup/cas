import torch
from matplotlib import pyplot as plt

def make_meshgrid(N1, N2, tkwargs):
    Xplt, Yplt = torch.meshgrid(
    torch.linspace(-3, 3, N1, **tkwargs), torch.linspace(-3, 3, N2, **tkwargs)
    )
    xplt = torch.stack(
        (
            torch.reshape(Xplt, (Xplt.shape[0] * Xplt.shape[1],)),
            torch.reshape(Yplt, (Yplt.shape[0] * Yplt.shape[1],)),
        ),
        dim=1,
    )
    return  Xplt, Yplt, xplt

def plot_3_graphs(X, feasible_inds_list, Xplt, Yplt, Z_list, title_list, constraints):
    # Check ready to plot
    for i in range(len(Z_list)):
        try: Z_list[i].numpy()
        except RuntimeError: Z_list[i] = Z_list[i].detach().numpy()
        except AttributeError: pass
        
    fig, ax = plt.subplots(1,3, figsize=(30, 7))
    for i in range(3):
        im = ax[i].contourf(Xplt, Yplt, Z_list[i], cmap="Blues", alpha=0.6)
        ax[i].plot(X[feasible_inds_list[i], 0], X[feasible_inds_list[i], 1], "og", label="Good")
        ax[i].plot(
            X[~feasible_inds_list[i], 0], X[~feasible_inds_list[i], 1], "or", label="Bad"
        )
        fig.colorbar(im, ax=ax[i])
        ind = 1
        for Xcor in X[5:]:
            ax[i].text(Xcor[0],Xcor[1],ind, size = 15)
            ind += 1
        ax[i].set_title(f"{title_list[i]}", fontsize = 20)
        ax[i].contour(Xplt, Yplt, Z_list[i], [constraints[i][1]], colors="k")
        ax[i].legend(loc=[0.83, 0.01])
    pass

def side_by_side(X, Xplt, Yplt, Z_list, N1, N2, title=None, feasible_inds=None, boundary=None):
    fig, ax = plt.subplots(1, 2, figsize = (20, 7))

    for i in range(2):
        im = ax[i].contourf(
            Xplt, Yplt, Z_list[i].reshape((N1,N2)), cmap="Blues", alpha=0.6)
        fig.colorbar(im, ax=ax[i])
        if title != None:
            ax[i].set_title(f"{title[i]}", fontsize = 20)
        if feasible_inds == None:
            ax[i].plot(X[:,0], X[:,1], "ko")
        else:
            ax[i].plot(X[feasible_inds, 0], X[feasible_inds, 1], "og", label="Good")
            ax[i].plot(
                X[~feasible_inds, 0], X[~feasible_inds, 1], "or", label="Bad"
            )
        if boundary != None:
            for j in range(len(boundary[i])):
                ax[i].contour(Xplt, Yplt, boundary[i][j], colors="k", linewidths=0.3)
            
    pass