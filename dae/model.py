import torch
import torch.nn as nn

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
