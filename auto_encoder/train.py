# ----------------------------------------------------------------------------------------------------
# IMPORTS --------------------------------------------------------------------------------------------
# ----------------------------------------------------------------------------------------------------

import yaml
import torch
import torchvision

import torch.nn as nn
import torch.optim as optim
import torchvision.datasets as datasets
import torchvision.transforms as transforms

from model import *
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter


# ----------------------------------------------------------------------------------------------------
# MODEL PREPARATION ----------------------------------------------------------------------------------
# ----------------------------------------------------------------------------------------------------

with open("configuration.yaml", "r") as file:
    config = yaml.safe_load(file)

# training parameters
n_epochs = config["training"]["n_epochs"]
lr = config["training"]["learning_rate"]
batch_size = config["training"]["batch_size"]

# model parameters
z = config["model"]["latent_dimension"]
input_dim = config["model"]["input_dimension"]

device = 'cuda' if torch.cuda.is_available() else 'cpu'

# model instanciation
model = AE(input_dim=input_dim, latent_dim=z)
model = model.to(device)

optimizer = optim.Adam(model.parameters(), lr=lr)
criterion = nn.MSELoss()

# data input pipeline
transforms = transforms.Compose(
    [transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))]
)

dataset = datasets.MNIST(root="dataset/", transform=transforms, download=True)
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

# tensorboard setup
writer_ae = SummaryWriter("runs/inputs_outputs")
writer_loss = SummaryWriter("runs/loss")
step = 0

# ----------------------------------------------------------------------------------------------------
# TRAINING LOOP --------------------------------------------------------------------------------------
# ----------------------------------------------------------------------------------------------------

for epoch in range(n_epochs):

    for batch_idx, (real, nbr) in enumerate(dataloader):

        input_images = real.view(-1, 784).to(device)
        batch_size = real.shape[0]
        
        optimizer.zero_grad()
        rebuilt_images = model(input_images)
        loss = criterion(input_images, rebuilt_images)
        loss.backward(retain_graph=True)
        optimizer.step()
        writer_loss.add_scalar("Training loss", loss, step)

        if batch_idx == 0:

            print(
                f"Epoch [{epoch}/{n_epochs}]"
                f"Loss : {loss:.4f}\n"
            )

            inputs = input_images.reshape(-1, 1, 28, 28)
            outputs = rebuilt_images.reshape(-1, 1, 28, 28)
            inputs_outputs = torch.cat((inputs, outputs), 0)
            grid = torchvision.utils.make_grid(inputs_outputs, normalize=True)
            
            writer_ae.add_image(
                "MNIST outputs images", grid, global_step=epoch
            )

        step += 1

