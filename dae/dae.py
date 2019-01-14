import torch
import torch.nn as nn
import torch.optim as optim
from torchvision.utils import save_image

from dae.model import Model
from dae.visualize import *

class DAE():
    def __init__(self, n_obs, num_epochs, batch_size, lr, save_iter, shape):
        self.num_epochs = num_epochs
        self.batch_size = batch_size
        self.lr = lr
        self.save_iter = save_iter
        self.shape = shape

        self.dae = Model(n_obs)

    def encode(self, x):
        return self.dae.encode(x)

    def decode(self, z):
        return self.dae.decode(z)

    def train(self, history):
        print('Training DAE...', end='', flush=True)

        optimizer = optim.Adam(self.dae.parameters(), lr=self.lr)

        for epoch in range(self.num_epochs):

            minibatches = history.get_minibatches(self.batch_size)
            for data in minibatches:

                out = self.dae(data)

                # calculate loss and update network
                loss = torch.pow(data - out, 2).mean()
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            if epoch == 0 or epoch % self.save_iter == self.save_iter - 1:
                pic = out.data.view(out.size(0), 1, self.shape[0], self.shape[1])
                save_image(pic, 'img/betaVae_' + str(epoch+1) + '_epochs.png')

            # plot loss
            update_viz(epoch, loss.item())

        print('DONE')
