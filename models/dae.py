import torch
import torch.nn as nn
import torch.optim as optim
from torchvision.utils import save_image

import numpy as np
from visdom import Visdom

viz = Visdom()

title = 'DAE Loss by Epoch'
win = None

def update_viz(epoch, loss):
    global win, title

    if win is None:
        title = title

        win = viz.line(
            X=np.array([epoch]),
            Y=np.array([loss]),
            win=title,
            opts=dict(
                title=title,
                fillarea=True
            )
        )
    else:
        viz.line(
            X=np.array([epoch]),
            Y=np.array([loss]),
            win=win,
            update='append'
        )


class Model(nn.Module):
    def __init__(self, n_obs):
        super(Model, self).__init__()

        self.encoder = nn.Sequential(
            nn.Linear(n_obs, 128),
            nn.ELU(),
            nn.Linear(128, 64),
            nn.ELU(),
            nn.Linear(64, 10)
        )

        self.decoder = nn.Sequential(
            nn.Linear(10, 64),
            nn.ELU(),
            nn.Linear(64, 128),
            nn.ELU(),
            nn.Linear(128, n_obs),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = x + torch.randn_like(x)
        return self.decoder(self.encoder(x))

    def encode(self, x):
        x = x + torch.randn_like(x)
        return self.encoder(x)

    def decode(self, z):
        return self.decoder(z)

class DAE():
    def __init__(self, n_obs, num_epochs, batch_size, lr):
        self.num_epochs = num_epochs
        self.batch_size = batch_size
        self.lr = lr

        self.dae = Model(n_obs)

    def encode(self, x):
        return self.dae.encode(x)

    def decode(self, z):
        return self.dae.decode(z)

    def train(self, history):
        print('Training DAE...', end='', flush=True)

        optimizer = optim.Adam(self.dae.parameters(), lr=self.lr)

        for epoch in range(self.num_epochs):

            minibatches = history.get_minibatches(self.batch_size, self.num_epochs)
            for data in minibatches:

                out = self.dae(data)

                # calculate loss and update network
                loss = torch.pow(data - out, 2).mean()
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            if epoch == 0 or epoch % 20 == 19:
                pic = out.data.view(out.size(0), 1, 28, 28)
                save_image(pic, 'img/dae_' + str(epoch+1) + '_epochs.png')

            # plot loss
            update_viz(epoch, loss.item())

        print('DONE')
