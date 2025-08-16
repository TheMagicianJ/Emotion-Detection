import numpy as np
from Layer import Layer


class Softmax(Layer):

    def __init__(self):
        super().__init__()

    def forward(self, input: np.ndarray) -> np.ndarray:

        #print(input)

        exp_input = np.exp(input)

        self.output = exp_input/ np.sum(exp_input)

        self.prediction = np.argmax(self.output)

        self.output = np.reshape(self.output, self.output.size)

        print(self.output.shape)

        return self.output
    
    def backward(self,error_grad: np.ndarray, learning_rate: float):

        return error_grad

        # n = np.size(self.output)

        # return np.dot((np.identity(n) - self.output.T) * self.output, error_grad)


    

