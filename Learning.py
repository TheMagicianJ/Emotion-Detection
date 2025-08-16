import numpy as np
from DataLoader import DataLoader


def predict(network: list, inputs: list) -> list:


    predictions = []

    for inp in inputs:

        inpt = inp

        for layer in network:

            output = layer.forward(inpt)
            inpt = output

                
        print(f"Prediction: {np.argmax(inpt)}")

        predictions.append(inpt)

    return predictions


def train(network: list, loss, lossPrime, dataLoader: DataLoader, learningRate : float = 0.01, epochs: int = 100):

    for e in range(epochs):

        loss_sum = 0

        for x,y in dataLoader:

            outputs = predict(network, x)

            loss_sum += loss(outputs, y)

            error_gradient = lossPrime(outputs, y)


            for layer in reversed(network):

                output = layer.backward(error_gradient, learningRate)

                error_gradient = output
        
        print(f"Epoch: {e + 1} --- Loss: {loss_sum}")




def test(network,x_test, y_test):

    correct = 0

    for x,y in zip(x_test, y_test):

        output = np.argmax(predict(network, x))

        if output == y:

            correct += 1

    print(f"Top-1 Accuracy: {correct/x_test.size}")



        


        
    





