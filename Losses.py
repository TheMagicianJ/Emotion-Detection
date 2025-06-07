import numpy as np

def logLoss(prediction, classes):
    
    loss =  ((1/classes) * prediction * np.log(prediction))/ classes

    return loss

def logLossPrime(prediction, classes):

    return

    