import yaml
import torch
import torchvision

import torch.optim as optim
import torchvision.datasets as datasets
import torchvision.transforms as transforms

from model import *
from torch.utils.data import DataLoader





with open('configuration.yaml', 'r') as file:
    config = yaml.safe_load(file)

# training parameters
n_epochs = config['training']['n_epochs']
lr = config['training']['learning_rate']
batch_size = config['training']['batch_size']

# model architecture
latent_dimension = config['model']['latent_dimension']
image_dimension = config['model']['image_dimension']

device = 'cuda' if torch.cuda.is_available() else 'cpu'

D = Discriminator(image_dim=image_dimension)
G = Generator(noise_dim=latent_dimension, image_dim=image_dimension)

D = D.to(device)
G = G.to(device)

noise = torch.randn((batch_size, latent_dimension))

transforms = transforms.Compose(
    [transforms.ToTensor(), transforms.Normalize((0.5,), (0.5))]
)

dataset = datasets.MNIST(root="dataset/", transform=transforms, download=True)
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

optimizer_D = optim.Adam(D.parameters(), lr=lr)
optimizer_G = optim.Adam(G.parameters(), lr=lr)

criterion = nn.BCELoss()

for epoch in range(n_epochs):

    for batch_idx, (real, _) in enumerate(dataloader):
        real_image = real.view(-1, 784).to(device)
        batch_size = real.shape[0]

        ### generator output ###
        noise = torch.rand(batch_size, z_dim).to(device)
        fake_image = G(noise)

        ### discriminator training ###
        disc_real = D(real_image).view(-1)
        disc_fake = D(fake_image).view(-1)
        loss_D_real = criterion(disc_real, torch.ones_like(disc_real))
        loss_D_fake = criterion(disc_fake, torch.ones_like(disc_fake))
        loss_D = (loss_D_real + loss_D_fake) / 2

        D.zero_grad()
        loss_D.backward()
        optimizer_D.step()

        ### generator training ###
