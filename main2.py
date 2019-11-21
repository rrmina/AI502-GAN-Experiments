import torch
import torch.nn as nn
import torch.optim as optim

import torchvision
from torchvision import transforms, datasets
import pickle as pkl

from models import Adversary, Generator
from loss import real_loss, fake_loss
import utils

# GLOBAL SETTINGS
BATCH_SIZE = 64
LATENT_DIM = 128
H = 28
W = 28
A_LR = 1e-3
G_LR = 1e-3
BETA_1 = 0.5
BETA_2 = 0.999
NUM_EPOCHS = 100

# Utils
SAMPLE_PATH = "train_sample.pkl"
SAVE_IMAGE_PATH = "results/"
SAVE_MODEL_PATH = "model/"

losses = []
samples = []
def train():
    # Device
    device = ("cuda" if torch.cuda.is_available() else "cpu")
    print("Device: {}".format(device))

    # Dataset and DataLoader
    transform = transforms.Compose([
        transforms.ToTensor()
    ])
    train_dataset = datasets.MNIST('data', train=True, transform=transform, download=True)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)

    # Load Networks
    a = Adversary(H, W).to(device)
    g = Generator(LATENT_DIM, H, W).to(device)

    # Optimizer Settings
    a_optim = optim.Adam(a.parameters(), lr=A_LR, betas=[BETA_1, BETA_2])
    g_optim = optim.Adam(g.parameters(), lr=G_LR, betas=[BETA_1, BETA_2])

    def generate_latent(batch_size, latent_dim):
        return torch.empty(batch_size, latent_dim).uniform_(-1,1).to(device)

    # Generate a fixed latent vector. This will be used 
    # in monitoring the improvement of generator network
    fixed_z = generate_latent(20, LATENT_DIM)

    def scale(tensor, mini, maxi):
        return tensor * (maxi - mini) + mini

    # Train Proper
    for epoch in range(1, NUM_EPOCHS+1):
        print("========Epoch {}/{}========".format(epoch, NUM_EPOCHS))

        for real_images, _ in train_loader:
            # Preprocess tensor
            batch_size = real_images.shape[0]               # Get the current batch size
            real_images = real_images.view(batch_size, -1)  # Reshape the tensor
            real_images = real_images.to(device)            # Move images to the appropriate device
            real_images = scale(real_images, -1, 1)         # Generator has Tanh output layer, so we need
                                                            # to rescale the real images to [-1,1]

            # Adversary Real Loss
            a_optim.zero_grad()                             # Zero-out gradients
            a_real_out = a(real_images)                     # Adversary forward pass
            a_real_loss = real_loss(a_real_out, smooth=True)# Compute loss according to real loss

            # Adversary Fake Loss                       
            z = generate_latent(batch_size, LATENT_DIM)     # Generate latent vectors
            fake_images = g(z)                              # Generator forward pass
            a_fake_out = a(fake_images)                     # Adversary forward pass
            a_fake_loss = fake_loss(a_fake_out, smooth=True)# Compute loss according to fake loss

            # Total Adversary Loss, Backprop and Gradient Descent
            a_loss = a_real_loss + a_fake_loss              # Total Loss 
            a_loss.backward()                               # Backprop
            a_optim.step()                                  # Gradient Descent

            # Generator Loss
            g_optim.zero_grad()                             # Zero-out gradients
            z = generate_latent(batch_size, LATENT_DIM)     # Generate latent vectors
            g_images = g(z)                                 # Generator forward pass
            a_g_out = a(g_images)                           # Adversary Forward Pass
            g_loss = real_loss(a_g_out)                     # Compute loss according to real loss
                                                            # Remember that the generator wants to fool the adversary
                                                            # Therefore we measure the performance of generator 
                                                            # wrt. how real the generated images are
                                                            # CONVERSELY, we may use the negative of fake_loss, instead

            # Generator Backprop and Gradient Descent
            g_loss.backward()
            g_optim.step()

        # Print Losses
        print("Adversary Loss: {} Generator Loss: {}".format(a_loss.item(), g_loss.item()))

        # Generate sample fake images after each epoch
        g.eval()
        sample_fake = g(fixed_z)
        sample_fake = sample_fake.view(-1, H, W)
        sample_images = utils.ttoi(sample_fake.clone().detach())
        samples.append(sample_images)
        g.train()

        # Save sample images
        sample_image_path = SAVE_IMAGE_PATH + "epoch" + str(epoch) + ".png"
        utils.save_samples_images(sample_images, sample_image_path)

    # Save sample fake images
    with open(SAMPLE_PATH, 'wb') as f:
        pkl.dump(sample_images, f)

    # Save the final model
    final_path = SAVE_MODEL_PATH + "final_model"
    torch.save(a.cpu().state_dict(), final_path + "_a.pth")
    torch.save(g.cpu().state_dict(), final_path + "_g.pth")

train()