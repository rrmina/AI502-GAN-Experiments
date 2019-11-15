import torch
import torch.nn as nn
import torch.optim as optim

import numpy as np

import torchvision
from torchvision import transforms, datasets

from models import Adversary, Generator
from loss import real_loss

# Global Settings
BATCH_SIZE = 64
Z_DIM = 100
H = 28
W = 28
GENERATOR_LR = 0.01
ADVERSARY_LR = 0.01
NUM_EPOCHS = 10

def main():
    print("Hello World")

    # Device
    device = ("cuda" if torch.cuda.is_available() else "cpu")

    # Dataset and Dataloader
    transform = transforms.Compose([
        transforms.ToTensor()
    ])
    train_dataset = datasets.MNIST('data', train=True, transform=transform, download=True)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    
    # Load networks
    g = Generator(Z_DIM, H, W)
    a = Adversary(H, W)

    # Optimizer Settings
    g_optim = optim.Adam(generator.parameters(), lr=GENERATOR_LR)
    a_optim = optim.Adam(adversary.parameters(), lr=ADVERSARY_LR)

    for epoch in range(1, NUM_EPOCHS+1):
        print("=======Epoch {}/{}========".format(epoch, NUM_EPOCHS))

        for real_images, _ in train_loader:
            # Get current batch size to reshape tensors
            curr_batch_size = images.shape[0]
            real_images = real_images.view(curr_batch_size, -1)

            # Pur the data to the appropriate device
            real_images = real_images.to(device)

            # Adversary Real Loss
            a_optim.zero_grad()
            a_real_out = a(real_images)
            a_real_loss = real_loss(a_real_out)

            # Adversary Fake Loss
            z = torch.empty(curr_batch_size, Z_DIM).uniform_(-1, 1).to(device)
            fake_images = g(z)
            a_fake_out = a(fake_images)
            a_fake_loss = fake_loss(a_fake_out)

            # Total Discriminator Loss, Backprop and Gradient Descent
            a_loss = a_real_loss + a_fake_loss
            a_loss.backward()
            a_optim.step()

            # Generator Loss
            g_optim.zero_grad()
            z = torch.empty(curr_batch_size, Z_DIM).uniform_(-1, 1).to(device)
            g_images = g(z)
            
            # Generator Training


main()