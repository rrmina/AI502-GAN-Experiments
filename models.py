import torch
import torch.nn as nn

class Generator(nn.Module):
    def __init__(self, LATENT_DIM, H, W):
        super(Generator, self).__init__()

        self.fc1 = nn.Linear(LATENT_DIM, 128)
        self.fc2 = nn.Linear(128, 256)
        self.fc3 = nn.Linear(256, 512)
        self.fc4 = nn.Linear(512, H*W)

        self.leaky_relu = nn.LeakyReLU(0.2)
        self.dropout = nn.Dropout(0.3)
        self.tanh = nn.Tanh()

    def forward(self, x):
        x = self.dropout(self.leaky_relu(self.fc1(x)))
        x = self.dropout(self.leaky_relu(self.fc2(x)))
        x = self.dropout(self.leaky_relu(self.fc3(x)))
        x = self.tanh(self.fc4(x))

        return x

class Adversary(nn.Module):
    def __init__(self, H=28, W=28):
        super(Adversary, self).__init__()
        
        self.fc1 = nn.Linear(H*W, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 128)
        self.fc4 = nn.Linear(128, 1)

        self.leaky_relu = nn.LeakyReLU(0.2)
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        x = self.dropout(self.leaky_relu(self.fc1(x)))
        x = self.dropout(self.leaky_relu(self.fc2(x)))
        x = self.dropout(self.leaky_relu(self.fc3(x)))
        x = self.fc4(x)

        return x