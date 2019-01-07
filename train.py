import torch
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import MNIST
from torchvision.utils import save_image

from models.dae import DAE
from models.beta_vae import BetaVAE
from visualize import *

# hyperparameters
num_epochs = 30
batch_size = 128
lr = 1e-4
beta = 2
save_iter = 10

# get images from MNIST database
dataset = MNIST('data', transform=transforms.ToTensor(), download=True)
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

# create DAE and its optimizer
dae = DAE()
dae_optimizer = optim.Adam(dae.parameters(), lr=lr)

# create ß-VAE and its optimizer
beta_vae = BetaVAE()
beta_vae_optimizer = optim.Adam(beta_vae.parameters(), lr=lr)

# train DAE
print('------------\ntraining DAE\n------------')
for epoch in range(num_epochs):

    # minibatch optimization with Adam
    for data in dataloader:
        img, labels = data

        # change the images to be 1D
        img = img.view(img.size(0), -1)

        # run images through DAE
        out = dae(img)

        # run one optimization step on the loss function
        loss = dae.loss(img, out)
        dae_optimizer.zero_grad()
        loss.backward()
        dae_optimizer.step()

    # save images periodically
    if epoch == 0 or epoch % save_iter == save_iter - 1:
        pic = out.data.view(out.size(0), 1, 28, 28)
        save_image(pic, 'img/dae_' + str(epoch+1) + '_epochs.png')

    # plot loss
    update_viz(epoch, loss.item(), 'DAE')
    print('%d / %d epochs' % (epoch + 1, num_epochs))

# train ß-VAE
print('--------------\ntraining ß-VAE\n--------------')
for epoch in range(num_epochs):

    # minibatch optimization with Adam
    for data in dataloader:
        img, labels = data

        # change the images to be 1D
        img = img.view(img.size(0), -1)

        # run images through ß-VAE
        out, mu, log_var = beta_vae(img)

        # run one optimization step on the loss function
        loss = beta_vae.loss(dae.encode(img).detach(), dae.encode(out), mu, log_var, beta)
        beta_vae_optimizer.zero_grad()
        loss.backward()
        beta_vae_optimizer.step()

    # save images periodically
    if epoch == 0 or epoch % save_iter == save_iter - 1:
        pic = out.data.view(out.size(0), 1, 28, 28)
        save_image(pic, 'img/betaVae_' + str(epoch+1) + '_epochs.png')

    # plot loss
    update_viz(epoch, loss.item(), 'ß-VAE')
    print('%d / %d epochs' % (epoch + 1, num_epochs))
