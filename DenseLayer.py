from Layer import Layer
import numpy as np

class Dense(Layer):
    
    def __init__(self, input_size, output_size):
        
        #Initialise random weights and a random bias
        self.weights = np.random.randn(output_size, input_size)
        self.bias = np.random.randn(output_size,1)



    def forward(self, input):

        self.input = input
        return np.dot(self.weights, self.input) + self.bias
    


    def backward(self, output_grad, learning_rate):

        # Updating the weights and biases.
        # The error gradient for the weights is weight X input(transposed)
        weights_gradient = np.dot(output_grad, self.input.T)
        self.weights -= learning_rate * weights_gradient
        self.bias -= learning_rate * output_grad

        # The error gradient with respect to the gradient is weights(transposed) x output_grad
        return np.dot(self.weights.T, output_grad)
