import numpy as np
from Layer import Layer


class Activation(Layer):

    def __init__(self, activation, activation_prime):

        self.activation = activation
        self.activation_prime = activation_prime

        

    def forward(self, input):
        self.input = input

        return self.activation(self.input)
    

    
    def backwards(self, output_gradient, learning_rate):
        
        return np.dot(output_gradient, self.activation_prime(self.input))



