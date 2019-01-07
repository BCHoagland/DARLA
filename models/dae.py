import torch
import torch.nn as nn
import torch.nn.functional as F

class DAE(nn.Module):
    def __init__(self):
        super(DAE, self).__init__()

        self.encoder = nn.Sequential(
            nn.Linear(28 * 28, 128),
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
            nn.Linear(128, 28 * 28),
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

    def loss(self, x, x_hat):
        return F.mse_loss(x_hat, x)
