import torch
import torch.nn as nn

class AE(nn.Module):

    def __init__(self, input_dim:int=784, latent_dim:int=64):
        super().__init__()
        self.encoder=nn.Sequential(
            nn.Linear(input_dim, input_dim//2),
            nn.ReLU(),
            nn.Linear(input_dim//2, latent_dim),
            nn.ReLU()
        )

        self.decoder=nn.Sequential(
            nn.Linear(latent_dim, input_dim//2),
            nn.ReLU(),
            nn.Linear(input_dim//2, input_dim),
            nn.ReLU()
        )

    def forward(self, x):
        return self.decoder(self.encoder(x))

    def encode(x):
        return self.encoder(x)

    def decode(x):
        return self.decoder(x)
