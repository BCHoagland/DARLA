import torch
import torch.nn as nn
import torch.nn.functional as F

class BetaVAE(nn.Module):
    def __init__(self):
        super(BetaVAE, self).__init__()

        self.encoder = nn.Sequential(
            nn.Linear(28 * 28, 128),
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
            nn.Linear(128, 28 * 28),
            nn.Sigmoid()
        )

    def forward(self, x):
        # encode
        x = self.encoder(x)
        mu = self.mu(x)
        log_var = self.log_var(x)
        # z = mu + (std_dev * eps), where eps ~ N(0,1)
        z = mu + torch.mul(torch.exp(log_var / 2), torch.randn_like(log_var))

        # decode
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

    def loss(self, x, x_hat, mu, log_var, beta):
        kl = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())
        kl /= mu.size(0) * 28 * 28              # mu.size(0) = batch size
        return F.mse_loss(x_hat, x) + (beta * kl)
