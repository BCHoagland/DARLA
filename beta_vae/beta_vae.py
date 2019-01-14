import torch
import torch.nn as nn
import torch.optim as optim
from torchvision.utils import save_image

from beta_vae.model import Model
from beta_vae.visualize import *

class BetaVAE():
    def __init__(self, n_obs, num_epochs, batch_size, lr, beta, save_iter, shape):
        self.num_epochs = num_epochs
        self.batch_size = batch_size
        self.lr = lr
        self.beta = beta
        self.save_iter = save_iter
        self.shape = shape

        self.n_obs = n_obs

        self.vae = Model(n_obs)

    def encode(self, x):
        return self.vae.encode(x)

    def decode(self, z):
        return self.vae.decode(z)

    def train(self, history, dae):
        print('Training ß-VAE...', end='', flush=True)

        def KL(mu, log_var):
            kl = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())
            kl /= mu.size(0) * self.n_obs
            return kl

        optimizer = optim.Adam(self.vae.parameters(), lr=self.lr)

        for epoch in range(self.num_epochs):

            minibatches = history.get_minibatches(self.batch_size)
            for data in minibatches:

                out, mu, log_var = self.vae(data)

                # calculate loss and update network
                loss = torch.pow(dae.encode(data) - dae.encode(out), 2).mean() + (self.beta * KL(mu, log_var))
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            if epoch == 0 or epoch % self.save_iter == self.save_iter - 1:
                pic = out.data.view(out.size(0), 1, self.shape[0], self.shape[1])
                save_image(pic, 'img/betaVae_' + str(epoch+1) + '_epochs.png')

            # plot loss
            update_viz(epoch, loss.item())

        print('DONE')
