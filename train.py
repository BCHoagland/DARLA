import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import MNIST

from models.dae import DAE
from models.beta_vae import BetaVAE
from history import History
from visualize import *

# hyperparameters
num_epochs = 100
batch_size = 128
lr = 1e-4
beta = 4
save_iter = 10

n_obs = 28 * 28

# create DAE and ß-VAE and their training history
dae = DAE(n_obs, num_epochs, batch_size, 1e-3)
beta_vae = BetaVAE(n_obs, num_epochs, batch_size, 1e-4, beta)
history = History()

# fill autoencoder training history with examples
print('Filling history...', end='', flush=True)
dataset = MNIST('data', transform=transforms.ToTensor(), download=True)
dataloader = DataLoader(dataset, batch_size=1, shuffle=True)
for data in dataloader:
    img, _ = data
    img = img.view(img.size(0), -1).numpy().tolist()
    history.store(img)
print('DONE')

# train DAE
dae.train(history)

# train ß-VAE
beta_vae.train(history, dae)
