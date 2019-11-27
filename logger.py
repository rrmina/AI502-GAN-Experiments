import numpy as np
import matplotlib.pyplot as plt

# Housekeeping Functions
def createLogger(names=[]):
    logs = {}
    for name in names:
        logs[name] = []

    return logs

def updateEpochLogger(curr_loggers, values=[]):
    # Record the loss of each batch
    curr_loggers["A_loss"].append(values[0])
    curr_loggers["G_loss"].append(values[1])
    curr_loggers["Acc"].append(values[2])
    return curr_loggers

def updateGlobalLogger(loggers, curr_loggers):
    # Record the Average Loss of each epoch
    # This is the average of batch losses in each epoch
    for key in loggers:
        loggers[key].append(np.mean(curr_loggers[key]))

    return loggers

def globalPlot(loggers):
    plotLoss(loggers)
    plotAcc(loggers)

def plotLoss(loggers):
    fig = plt.figure(figsize=(10,10))
    plt.plot(loggers["A_loss"], label="Adversary Loss")
    plt.plot(loggers["G_loss"], label="Generator Loss")
    plt.title("Training Losses")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.show()

def plotAcc(loggers):
    fig = plt.figure(figsize=(10,10))
    plt.plot(loggers["Acc"], label="Adversary Accuracy")
    plt.title("Adversary Acuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.show()