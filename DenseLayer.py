from Layer import Layer
import numpy as np

# To-do: Annotations

class Dense(Layer):
    
    def __init__(self, input_size, output_size):
        
        self.weights = np.random.randn(output_size, input_size)
        self.bias = np.random.randn(output_size,1)



    def forward(self, input):

        self.input = input
        return np.dot(self.weights, self.input) + self.bias
    


    def backward(self, output_grad, learning_rate):

        weights_gradient = np.dot(output_grad, self.input.T)
        self.weights -= learning_rate * weights_gradient
        self.bias -= learning_rate * output_grad

        return np.dot(self.weights.T, output_grad)
