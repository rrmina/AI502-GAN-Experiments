import torch
import torch.nn as nn

class Generator(nn.Module):
    def __init__(self, LATENT_DIM, H, W):
        super(Generator, self).__init__()

        self.fc1 = nn.Linear(LATENT_DIM, 128)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(128, H*W)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu1(x)
        x = self.fc2(x)

        return x

class Adversary(nn.Module):
    def __init__(self, H=28, W=28):
        super(Adversary, self).__init__()
        
        self.fc1 = nn.Linear(H*W, 128)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(128, 1)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu1(x)
        x = self.fc2(x)

        return x

#def AdversarialModel(nn.Module):
#    def __init__(self):
#        super(AdversaryModel, self).__init__()
        
        # Global Settings and Hyperparameters