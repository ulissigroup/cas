from matplotlib import pyplot as plt


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