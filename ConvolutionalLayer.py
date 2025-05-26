import numpy as np
from Layer import Layer
from scipy import signal

class ConvultionLayer(Layer):



    def __init__(self, input_shape, kernel_size, depth):

        self.depth = depth
        self.input_shape = input_shape
        input_width, input_height, input_depth = input_shape
        self.input_depth = input_depth
        self.kernel_size = kernel_size
        self.output_shape = (depth, (input_height - kernel_size + 1), (input_width - kernel_size + 1))
        self.kernels_shape = (depth, input_depth, kernel_size, kernel_size)
        self.kernels = np.random.randn(self.kernel_shape)
        self.biases = np.random.randn(self.output_shape)



    def forward(self, input):

        self.input = input
        self.output = np.copy(self.biases)

        for i in range(self.depth):

            for j in range(self.input_depth):

                self.output[i] += signal.correlate2d(self.input[j], self.kernels[i, j], "valid")
        
        return self.output
    


    def backwords(self, output_gradient, learning_rate):

        kernels_gradient = np.zeros(self.kernels_shape)
        input_gradient = np.zeros(self.input_shape)

        for i in range(self.depth):
            
            for j in range(self.input_depth):

                kernels_gradient[i, j] = signal.correlated2d(self.input[j], output_gradient[i], "valid")
                input_gradient[j] += signal.convolve2d(output_gradient, self.kernels[i, j], "full")

        self.kernels -= learning_rate * kernels_gradient
        self.biases -= learning_rate * output_gradient

        return input_gradient

