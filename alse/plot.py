import torch
from matplotlib import cm
import matplotlib.pyplot as plt
from alse.test_function.fxns import yf
import gpytorch
from PIL import Image
import glob
def plot(gp_model_hills, gp_model_circle, gp_model_class, X, Yhills, Yhills, Yyf):
    N1, N2 = 200, 200
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
    yplt_hills = hills(xplt)
    Zplt_hills = torch.reshape(yplt_hills, (N1, N2))
    yplt_circle = circle(xplt)
    Zplt_circle = torch.reshape(yplt_circle, (N1, N2))
    yplt_yf = yf(xplt)
    Zplt_yf = torch.reshape(yplt_yf, (N1, N2)) 

    fig, ax = plt.subplots(2,3, figsize=(30, 15))
    plot1 = ax[0][0].contourf(Xplt.cpu(), Yplt.cpu(), Zplt_hills.cpu(), levels=np.linspace(-2, 2, 15), cmap="Blues", alpha=0.6)
    ax[0][0].plot(X[feasible_inds_hills, 0].cpu(), X[feasible_inds_hills, 1].cpu(), "sg", label="Feasible")
    ax[0][0].plot(
        X[~feasible_inds_hills, 0].cpu(), X[~feasible_inds_hills, 1].cpu(), "sr", label="Infeasible"
    )
    ax[0][0].scatter(X.cpu()[:5, 0], X.cpu()[:5, 1], marker = 'o', s=100, color = 'k')
    ind = 1
    for i in X[5:]:
        ax[0][0].text(i[0],i[1],ind, size = 15)
        ind += 1
    ax[0][0].legend(loc=[0.7, 0.05])
    ax[0][0].set_title('Hills function', fontsize=20)

    ax[0][0].contour(Xplt.cpu(), Yplt.cpu(), Zplt_hills.cpu(), [constraints[0][1]], colors="k")
    fig.colorbar(plot1, ax=ax[0])

    plot2 = ax[0][1].contourf(Xplt.cpu(), Yplt.cpu(), Zplt_circle.cpu(), levels=np.linspace(0, 30, 15), cmap="Blues", alpha=0.6)
    ax[0][1].plot(X[feasible_inds_circle, 0].cpu(), X[feasible_inds_circle, 1].cpu(), "sg", label="Feasible")
    ax[0][1].plot(
        X[~feasible_inds_circle, 0].cpu(), X[~feasible_inds_circle, 1].cpu(), "sr", label="Infeasible"
    )
    ax[0][1].scatter(X.cpu()[:5, 0], X.cpu()[:5, 1], marker = 'o', s=100, color = 'k')
    ax[0][1].contour(Xplt.cpu(), Yplt.cpu(), Zplt_circle.cpu(), [constraints[1][1]], colors="k")
    ind = 1
    for i in X[5:]:
        ax[0][1].text(i[0],i[1],ind, size = 15)
        ind += 1
    ax[0][1].legend(loc=[0.7, 0.05])
    ax[0][1].set_title('Circle function', fontsize=20)
    fig.colorbar(plot2, ax=ax[0][1], location='right')

    plot2 = ax[0][2].contourf(Xplt.cpu(), Yplt.cpu(), Zplt_yf.cpu(), cmap="Blues", alpha=0.6)
    ax[0][2].contour(Xplt.cpu(), Yplt.cpu(), Zplt_yf.cpu(), colors="k", linewidths=0.3)
    ax[0][2].scatter(X.cpu()[:5, 0], X.cpu()[:5, 1], marker = 'o', s=100, color = 'k')

    ax[0][2].plot(X[feasible_inds_yf, 0].cpu(), X[feasible_inds_yf, 1].cpu(), "sg", label="Feasible")
    ax[0][2].plot(
        X[~feasible_inds_yf, 0].cpu(), X[~feasible_inds_yf, 1].cpu(), "sr", label="Infeasible"
    )
    ind = 1
    for i in X[5:]:
        ax[0][2].text(i[0],i[1],ind, size = 15)
        ind += 1
    ax[0][2].set_title('Classification function', fontsize=20)
    fig.colorbar(plot2, ax=ax[2], location='right')


    pos_yplt_hills = gp_model_hills(xplt).loc
    pos_yplt_circle = gp_model_circle(xplt).loc

    pos_Zplt_hills = torch.reshape(pos_yplt_hills, (N1, N2))
    pos_Zplt_circle = torch.reshape(pos_yplt_circle, (N1, N2))

    pos_yplt_yf = gp_model_class(xplt.float())
    pred_means = pos_yplt_yf.loc
    pred_samples = pos_yplt_yf.sample(torch.Size((256,))).exp()
    probabilities = (pred_samples / pred_samples.sum(-2, keepdim=True)).mean(0)
    probabilities[0] = probabilities[0]*(probabilities[0].max()-probabilities[0].min())
    probabilities[1] = probabilities[1]/probabilities[1].max()
    pos_Zplt_yf = probabilities.max(0)[1].reshape((N1,N2))

    h1 = ax[1][0].contourf(Xplt.cpu().detach(), Yplt.cpu().detach(), pos_Zplt_hills.cpu().detach(), levels=np.linspace(-2, 2, 15), cmap="Blues", alpha=0.6)
    ax[1][0].contour(Xplt.cpu().detach(), Yplt.cpu().detach(), pos_Zplt_hills.cpu().detach(), [0.3], colors="k")
    fig.colorbar(h1, ax=ax[1][0])

    h2 = ax[1][1].contourf(Xplt.cpu().detach(), Yplt.cpu().detach(), pos_Zplt_circle.cpu().detach(), levels=np.linspace(0, 30, 15), cmap="Blues", alpha=0.6)
    ax[1][1].contour(Xplt.cpu().detach(), Yplt.cpu().detach(), pos_Zplt_circle.cpu().detach(), [20], colors="k")
    fig.colorbar(h2, ax=ax[1])

    h3 = ax[1][2].contourf(Xplt.cpu().detach(), Yplt.cpu().detach(), pos_Zplt_yf.cpu().detach(), cmap="Blues", alpha=0.6)
    ax[1][2].contour(Xplt.cpu().detach(), Yplt.cpu().detach(), pos_Zplt_yf.cpu().detach(), colors="k", linewidths=0.3)
    fig.colorbar(h3, ax=ax[1][2])

def identify_samples_which_satisfy_constraints(X):
    """
    Takes in values (a1, ..., ak, o) and returns (a1, ..., ak, o)
    True/False values, where o is the number of outputs.
    """
    successful = torch.ones(X.shape).to(X)
    for model_index in range(X.shape[-1]):
        these_X = X[..., model_index]
        # direction, value = constraints[model_index]
        # successful[..., model_index] = (
        #     these_X < value if direction == "lt" else these_X > value
        # )
        successful[..., model_index] = these_X == 1
    return successful

def plot_acq_pos_gif(model, queried_pts_x, queried_pts_y, num_init_points, lb, ub, tkwargs = None):

    N1, N2 = 50, 50
    Xplt, Yplt = torch.meshgrid(
        torch.linspace(lb, ub, N1, **tkwargs), torch.linspace(lb, ub, N2, **tkwargs)
    )
    xplt = torch.stack(
        (
            torch.reshape(Xplt, (Xplt.shape[0] * Xplt.shape[1],)),
            torch.reshape(Yplt, (Yplt.shape[0] * Yplt.shape[1],)),
        ),
        dim=1,
    )
    yplt = yf(xplt)
    Zplt = torch.reshape(yplt[:, 0], (N1, N2))

    fig, (ax, ax1) = plt.subplots(1, 2, figsize=(16, 6))
    h1 = ax.contourf(Xplt.cpu(), Yplt.cpu(), Zplt.cpu(), 20, cmap="Blues", alpha=0.6)
    fig.colorbar(h1, ax = ax)
    ax.contour(Xplt.cpu(), Yplt.cpu(), Zplt.cpu(), [0, 1], colors="k")

    feasible_inds = (
        identify_samples_which_satisfy_constraints(queried_pts_y).prod(dim=-1).to(torch.bool)
    )
    ax.plot(queried_pts_x[feasible_inds, 0].cpu(), queried_pts_x[feasible_inds, 1].cpu(), "sg", label="Feasible")
    ax.plot(
        queried_pts_x[~feasible_inds, 0].cpu(), queried_pts_x[~feasible_inds, 1].cpu(), "sr", label="Infeasible"
    )
    ax.scatter(
        queried_pts_x.cpu()[:num_init_points, 0],
        queried_pts_x.cpu()[:num_init_points, 1],
        marker="o",
        s=120,
        color="k",
        label="Training",
    )

    ind = 1
    for i in queried_pts_x[num_init_points:]:
        ax.text(i[0], i[1], ind, size=15)
        ind += 1
    ax.legend()
    ax.set_title("$f_1(x)$")  # Recall that f1(x) = f2(x)
    ax.set_xlabel("$x_1$")
    ax.set_ylabel("$x_2$")
    # ax.set_aspect("equal", "box")


    model.eval()

    with gpytorch.settings.fast_pred_var(), torch.no_grad():
        test_dist = model(xplt.float())
        pred_means = test_dist.loc
    pred_samples = test_dist.sample(torch.Size((256,))).exp()
    probabilities = (pred_samples / pred_samples.sum(-2, keepdim=True)).mean(0)
    ax1.contourf(Xplt.cpu(), Yplt.cpu(), pred_means.max(0)[1].reshape((N1,N2)))
    ax1.contour(Xplt.cpu(), Yplt.cpu(), Zplt.cpu(), [0, 1], colors="k")

    ax1.set_title("$posterior$") 
    ax1.set_xlabel("$x_1$")
    ax1.set_ylabel("$x_2$")
    fig_id = queried_pts_x.shape[0]-num_init_points
    plt.savefig(f"results/iter_{fig_id:02d}.png")

    # Create the frames
    frames = []
    imgs = sorted(glob.glob("results/*.png"))
    for i in imgs:
        new_frame = Image.open(i)
        frames.append(new_frame)
    
    # Save into a GIF file that loops forever
    frames[0].save('results/result.gif', format='GIF',
                append_images=frames[1:],
                save_all=True,
                duration=600, loop=0)