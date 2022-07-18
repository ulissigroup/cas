import torch, gpytorch
from alse.fit_model import fit_gp_class, fit_gp_reg
import matplotlib as plt
import numpy as np

def fit_all_models(x, N1, N2, X, rand_y_hills, rand_y_circle, rand_y_class):
    model_hills = fit_gp_reg(X, rand_y_hills)
    model_circle = fit_gp_reg(X, rand_y_circle)
    model_class = fit_gp_class(X, rand_y_class, lr=0.05, iter=300)

    model_class.eval()
    with gpytorch.settings.fast_pred_var(), torch.no_grad():
        test_dist = model_class(x.float())
        pred_means = test_dist.loc #Save this in case we need to switch how we evaluate boundary

    pred_samples = test_dist.sample(torch.Size((256,))).exp()
    probabilities = (pred_samples / pred_samples.sum(-2, keepdim=True)).mean(0)
    probabilities[0] = (probabilities[0]-probabilities[0].min())/(probabilities[0].max()-probabilities[0].min())
    probabilities[1] = (probabilities[1]-probabilities[1].min())/(probabilities[1].max()-probabilities[1].min())

    ###################################################################
    # fig, ax = plt.subplots(1, 2, figsize = (20, 7))

    # for i in range(2):
    #     im = ax[i].contourf(
    #         X.numpy(), Y.numpy(), pred_means[i].numpy().reshape(N1,N2),
    #     cmap="Blues", alpha=0.6)
    #     fig.colorbar(im, ax=ax[i])
    #     ax[i].plot(rand_x[:,0].numpy(), rand_x[:,1].numpy(), "ko")
    #     ax[i].set_title("Logits: Class " + str(i), fontsize = 20)
    ###################################################################
    # fig, ax = plt.subplots(1, 2, figsize = (20, 7))

    # levels = np.linspace(0, 1.05, 20)
    # for i in range(2):
    #     im = ax[i].contourf(
    #         X.numpy(), Y.numpy(), probabilities[i].numpy().reshape(N1,N2), levels=levels,
    #     cmap="Blues", alpha=0.6)
    #     fig.colorbar(im, ax=ax[i])
    #     ax[i].plot(rand_x[:,0].numpy(), rand_x[:,1].numpy(), "ko")
    #     ax[i].set_title("Probabilities: Class " + str(i), fontsize = 20)
    ##################################################################
    predicted_hills = model_hills(x).loc.reshape(N1,N2)
    predicted_circle = model_circle(x).loc.reshape(N1,N2)
    predicted_yf = probabilities.max(0)[1].reshape(N1,N2)
    return predicted_hills, predicted_circle, predicted_yf