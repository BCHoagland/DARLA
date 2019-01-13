import torch
import torch.nn as nn

class Model(nn.Module):
    def __init__(self, n_obs):
        super(Model, self).__init__()

        self.encoder = nn.Sequential(
            nn.Linear(n_obs, 128),
            nn.ELU(),
            nn.Linear(128, 64),
            nn.ELU()
        )

        self.mu = nn.Sequential(
            nn.Linear(64, 10)
        )

        self.log_var = nn.Sequential(
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
        x = self.encoder(x)
        mu = self.mu(x)
        log_var = self.log_var(x)

        z = mu + torch.mul(torch.exp(log_var / 2), torch.randn_like(log_var))
        x_hat = self.decoder(z)

        return x_hat, mu, log_var

    def encode(self, x):
        x = self.encoder(x)
        mu = self.mu(x)
        log_var = self.log_var(x)
        z = mu + torch.mul(torch.exp(log_var / 2), torch.randn_like(log_var))
        return z

    def decode(self, z):
        return self.decoder(z)
