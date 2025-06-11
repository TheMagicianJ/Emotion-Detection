import numpy as np
from Layer import Layer

#TO-DO: Annontations

class Activation(Layer):

    def __init__(self, activation, activation_prime):

        # Activation function for propragation.
        self.activation = activation
        # Backwards propagation activation is the inverse of the forward which means they have to be individually defined.
        self.activation_prime = activation_prime

        
    def forward(self, input):
        self.input = input

        return self.activation(self.input)
    

    
    def backwards(self, output_gradient, learning_rate):
        
        return np.dot(output_gradient, self.activation_prime(self.input))



