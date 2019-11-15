import torch
import torch.nn as nn

def real_loss(x, smooth=False):
    # Get the number of samples
    batch_size = x.shape[0]

    # Label Smoothing
    if (smooth):
        labels = torch.ones(batch_size) * 0.9
    else:
        labels = torch.ones(batch_size)

    # Move to the appropriate device
    device = ("cuda" if torch.cuda.is_available() else "cpu")
    labels = labels.to(device)

    # Compute Loss
    criterion = nn.BCEWithLogitsLoss()
    return criterion(x.squeeze(), labels)

def fake_loss(x, smooth=False):
    # Get the number of samples
    batch_size = x.shape[0]

    # Label Smoothing
    if (smooth):
        labels = torch.zeros(batch_size) * 0.9
    else:
        labels = torch.zeros(batch_size)

    # Move the data to the appropriate device
    device = ("cuda" if torch.cuda.is_available() else "cpu")
    labels = labels.to(device)

    # Compute Losss
    criterion = nn.BCEWithLogitsLoss()
    return criterion(x.squeeze(), labels)