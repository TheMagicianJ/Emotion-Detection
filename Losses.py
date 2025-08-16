import numpy as np

def logLoss(predictions: list, true_ys: list):

    loss = 0

    for i in range(len(true_ys)):

        loss += -true_ys[i] * np.log(predictions[i])

    loss = np.sum(loss)

    return loss


def logLossPrime(predictions: list, true_ys: list) -> np.ndarray:


    preds = np.array(predictions)
    trues = np.array(true_ys)

    print(f"prediction shape: {preds.shape}")
    print(f"trues shape: {trues.shape}")

    error = np.sum((preds - trues), axis = 0)/ len(predictions)

    error = error.reshape(len(predictions[0]),1)

    return error

     