import numpy as np
from Layer import Layer


class MaxPooling(Layer):

    def __init__(self, window_size, stride) :

        self.window = (window_size,window_size)
        self.stride = stride
        self.max_positions = ()


    def forward(input):

        # (input size - pooling window size + 2 * padding) / stride ) + 1

        return
    
    def backward(self, output_grad,):

        # Gradient only matter only on the positions where the max was found everything else is left the same so pad out with zero.

        return
        

