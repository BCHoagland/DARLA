import gym
import torch
import torch.optim as optim

from dae.dae import DAE
from beta_vae.beta_vae import BetaVAE
from history import History
from visualize import *

shape = (125, 80)
n_obs = shape[0] * shape[1]

# returns 80 by 80 image
def preprocess(img):
    processed = img[:,:,0]
    processed = processed[::2,::2]
    processed = processed / 255
    t = torch.FloatTensor(processed)
    t = t.view(n_obs)
    return t

# hyperparameters
num_epochs = 200
batch_size = 128
beta = 4
save_iter = 50

# create DAE and ß-VAE and their training history
dae = DAE(n_obs, num_epochs, batch_size, 1e-3, save_iter, shape)
beta_vae = BetaVAE(n_obs, num_epochs, batch_size, 1e-4, beta, save_iter, shape)
history = History()

# fill autoencoder training history with examples
print('Filling history...', end='', flush=True)
env = gym.make('AirRaid-v0')

s = env.reset()
for step in range(4000):
    history.store(preprocess(s))
    s, _, done, _ = env.step(env.action_space.sample())
    if done:
        s = env.reset()
print('DONE')

# train DAE
dae.train(history)

# train ß-VAE
beta_vae.train(history, dae)
