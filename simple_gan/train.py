# ----------------------------------------------------------------------------------------------------
# IMPORTS --------------------------------------------------------------------------------------------
# ----------------------------------------------------------------------------------------------------

import yaml
import torch
import torchvision

import torch.optim as optim
import torchvision.datasets as datasets
import torchvision.transforms as transforms

from models import *
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter


# ----------------------------------------------------------------------------------------------------
# MODEL PREPARATION ----------------------------------------------------------------------------------
# ----------------------------------------------------------------------------------------------------

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

#model instanciation
D = Discriminator(image_dim=image_dimension)
G = Generator(noise_dim=latent_dimension, image_dim=image_dimension)

D = D.to(device)
G = G.to(device)

fx_noise = torch.randn((batch_size, latent_dimension))

transforms = transforms.Compose(
    [transforms.ToTensor(), transforms.Normalize((0.5,), (0.5))]
)

dataset = datasets.MNIST(root="dataset/", transform=transforms, download=True)
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

optimizer_D = optim.Adam(D.parameters(), lr=lr)
optimizer_G = optim.Adam(G.parameters(), lr=lr)

criterion = nn.BCELoss()

# writers to track results on tensorboard
writer_fake = SummaryWriter("runs/simple_gan/fake")
writer_real = SummaryWriter("runs/simple_gan/real")
writer_loss = SummaryWriter("runs/simple_gan/loss")
step = 0


# ----------------------------------------------------------------------------------------------------
# TRAINING LOOP --------------------------------------------------------------------------------------
# ----------------------------------------------------------------------------------------------------

for epoch in range(n_epochs):

    for batch_idx, (real, _) in enumerate(dataloader):
        real_image = real.view(-1, 784).to(device)
        batch_size = real.shape[0]

        ### generator output ###
        noise = torch.randn(batch_size, latent_dimension).to(device)
        fake_image = G(noise)

        ### discriminator training ###
        disc_real = D(real_image).view(-1)
        disc_fake = D(fake_image).view(-1)
        loss_D_real = criterion(disc_real, torch.ones_like(disc_real))
        loss_D_fake = criterion(disc_fake, torch.zeros_like(disc_fake))
        loss_D = (loss_D_real + loss_D_fake) / 2

        D.zero_grad()
        loss_D.backward(retain_graph=True)
        optimizer_D.step()

        ### generator training ###
        output = D(fake_image).view(-1)
        loss_G = criterion(output, torch.ones_like(output))
        G.zero_grad()
        loss_G.backward()
        optimizer_G.step()


        ### tensorboard ###
        if batch_idx == 0:
            
            print(
                f"Epoch [{epoch}/{n_epochs}]" 
                f"Loss D : {loss_D:.4f} Loss G : {loss_G:.4f}\n"
            )

            writer_loss.add_scalar("Generarator Loss", loss_G, epoch)
            writer_loss.add_scalar("Discriminator Loss", loss_D, epoch)

            with torch.no_grad():
                fake = G(fx_noise).reshape(-1, 1, 28, 28)
                real = real_image.reshape(-1, 1, 28, 28)
                grid_fake = torchvision.utils.make_grid(fake, normalize=True)
                grid_real = torchvision.utils.make_grid(real, normalize=True)

                writer_fake.add_image(
                    "MNIST fake images", grid_fake, global_step=step
                )

                writer_real.add_image(
                    "MNIST real images", grid_real, global_step=step
                )
                step += 1



