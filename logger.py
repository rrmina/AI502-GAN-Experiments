import numpy as np

# Housekeeping Functions
def createLogger(names=[]):
    logs = {}
    for name in names:
        logs[name] = []

    return logs

def updateEpochLogger(curr_loggers, values=[]):
    # Record the loss of each batch
    curr_loggers["A_loss"].append(values[0].item())
    curr_loggers["G_loss"].append(values[1].item())
    curr_loggers["Acc"].append(values[2].item())
    return curr_loggers

def updateGlobalLogger(loggers, curr_loggers):
    # Record the Average Loss of each epoch
    # This is the average of batch losses in each epoch
    for key in loggers:
        loggers[key].append(np.mean(curr_loggers[key]))

    return loggers