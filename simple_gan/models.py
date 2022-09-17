import torch
import torch.nn as nn

class Discriminator(nn.Module):

    def __init__(self, image_dim:tuple):

        super().__init__()
        self.discriminator = nn.Sequential(
            nn.Linear(image_dim, 128),
            nn.LeakyReLU(0.01),
            nn.Linear(128, 1),
            nn.Sigmoid(),
        )

    def forward(self, x):

        return self.discriminator(x)

class Generator(nn.Module):

    def __init__(self, noise_dim:int, image_dim:tuple):

        super().__init__()
        self.generator = nn.Sequential(
            nn.Linear(noise_dim, 256),
            nn.LeakyReLU(0.01),
            nn.Linear(256, image_dim),
            nn.Tanh(),
        )

    def forward(self, x):

        return self.generator(x)