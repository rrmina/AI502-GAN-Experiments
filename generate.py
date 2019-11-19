import torch
import torch.nn as nn
from models import Adversary, Generator
from utils import show, concatenate_images, ttoi, save_samples_images
import matplotlib.pyplot as plt
import numpy as np

# General settings
ADVERSARY_MODEL_PATH = "model/final_model_a.pth"
GENERATOR_MODEL_PATH = "model/final_model_g.pth"

# Hyperparameters
H = 28
W = 28
LATENT_DIM = 128

# Number of Images
NUM_IMAGES = 20

def generate_latent(batch_size, latent_dim, device):
    return torch.empty(batch_size, latent_dim).uniform_(-1,1).to(device)

def generate_images():
    # Device
    device = ("cuda" if torch.cuda.is_available() else "cpu")

    # Instantiate Network and Load Pre-trained models
    g = Generator(LATENT_DIM, H, W)
    g.load_state_dict(torch.load(GENERATOR_MODEL_PATH))
    g = g.to(device)

    with torch.no_grad():
        torch.cuda.empty_cache()
        print("Generating Images")
        
        z = generate_latent(NUM_IMAGES, LATENT_DIM, device) # Latent Variables
        generated_tensor = g(z)
        generated_tensor = generated_tensor.view(-1, H, W)

        generated_images = ttoi(generated_tensor)
        #plt.imshow(generated_images[0])
        #plt.show()
        save_samples_images(generated_images, "sample.png")
        image = plt.imread("sample.png")
        plt.imshow(image[:28, :28], cmap="gray")
        plt.show()


generate_images()