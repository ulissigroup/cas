import numpy as np
from gpytorch.likelihoods import DirichletClassificationLikelihood
from gp_model import DirichletGPModel
import math
import torch
import numpy as np
import gpytorch
from matplotlib import pyplot as plt

class Alse():

    def __init__(self, data_x, data_y, d1, d2):

    self.train_x = data_x
    self.train_y = data_y
    self.test_x_mat, self.test_y_mat = np.meshgrid(d1, d2)
    self.test_x_mat, self.test_y_mat = torch.Tensor(self.test_x_mat), torch.Tensor(self.test_y_mat)
    self.test_x = torch.cat((self.test_x_mat.view(-1,1), self.test_y_mat.view(-1,1)), dim=1)

    def run(self):
        return

    def init_gp(self):
        self.likelihood = DirichletClassificationLikelihood(self.train_y, learn_additional_noise=True)
        self.gp = DirichletGPModel(self.train_x, likelihood.transformed_targets, likelihood, num_classes=likelihood.num_classes)
        return
    
    def predict(self):
        self.gp.eval()
        self.likelihood.eval()
        with gpytorch.settings.fast_pred_var(), torch.no_grad():
            test_dist = self.gp(self.test_x)
            self.pred_means = test_dist.loc
        self.plot_prediction()
    
    def plot_predictions(self):
        fig, ax = plt.subplots(1, 3, figsize = (15, 5))
        for i in range(3):
            im = ax[i].contourf(
                self.test_x_mat.numpy(), self.test_y_mat.numpy(), self.pred_means[i].numpy().reshape((20,20))
            )
            fig.colorbar(im, ax=ax[i])
            ax[i].set_title("Logits: Class " + str(i), fontsize = 20)

    def acquisition(self):
        return
    
    def find_next_step(self):
        return
    

    def hpo(self):
        return
