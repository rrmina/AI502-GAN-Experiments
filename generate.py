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
NUM_IMAGES = 100
NUM_ROWS_SAMPLE = 10

###########################################################
# Helper Functions for producing latent variables
###########################################################
def generate_latent_uniform(batch_size, latent_dim, device):
    return torch.empty(batch_size, latent_dim).uniform_(-1,1).to(device)

def generate_latent_normal(batch_size, latent_dim, device):
    return torch.empty(batch_size, latent_dim).normal_(0,1).to(device)

def generate_latent_sweep(batch_size, latent_dim, device):

    # Placeholder Tensor
    z = torch.empty(batch_size, latent_dim).to(device)

    # Sweep Values
    step = 2 / batch_size
    sweep = np.arange(-1, 1, step)
    sweep = torch.from_numpy(sweep).to(device).float()

    # Replace placeholder with sweep values
    for i in range(128):
        z.T[i] = sweep
    
    return z

def scale_back(tensor):
    return (tensor+1) / 2

############################
# Generate Function
############################
def generate_images(latent="uniform"):
    # Device
    device = ("cuda" if torch.cuda.is_available() else "cpu")

    # Instantiate Network and Load Pre-trained models
    g = Generator(LATENT_DIM, H, W)
    g.load_state_dict(torch.load(GENERATOR_MODEL_PATH))
    g = g.to(device)

    # Generate latent vectors
    if (latent=="uniform"):
        z = generate_latent_uniform(NUM_IMAGES, LATENT_DIM, device)
    elif (latent == "normal"):
        z = generate_latent_normal(NUM_IMAGES, LATENT_DIM, device)
    elif (latent == "sweep"):
        z = generate_latent_sweep(NUM_IMAGES, LATENT_DIM, device)

    with torch.no_grad():
        torch.cuda.empty_cache()
        print("Generating Images. Latent: {} Device: {}".format(latent, device))
        
        # Generate image tensor | Transform tensor to numpy arrays
        generated_tensor = g(z)
        generated_tensor = generated_tensor.view(-1, H, W)
        generated_images = ttoi(generated_tensor)
        generated_images = scale_back(generated_images)

        # Save 
        filename = "sample_" + latent + ".png" 
        save_samples_images(generated_images, filename, NUM_IMAGES, NUM_ROWS_SAMPLE, H, W)

        # and show images
        concat_images = concatenate_images(generated_images, NUM_IMAGES, NUM_ROWS_SAMPLE) / 255
        concat_images = concat_images.clip(0,1) # Need to clip!ename)
        show(concat_images)

generate_images("uniform")
generate_images("sweep")