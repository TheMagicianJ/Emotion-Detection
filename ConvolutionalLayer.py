import numpy as np
from Layer import Layer
from scipy import signal

class ConvultionLayer(Layer):



    def __init__(self, input_shape, kernel_size, depth):


        # kernel_Size represents the dimensions of each matrix in a kernel
        # depth represents how many kernels

        self.depth = depth
        self.input_shape = input_shape
        input_width, input_height, input_depth = input_shape
        self.input_depth = input_depth
        self.kernel_size = kernel_size

        # The amount of output matrices should be the same as number of kernels
        self.output_shape = (depth, (input_height - kernel_size + 1), (input_width - kernel_size + 1))

        # The depth of the kernels 
        self.kernels_shape = (depth, input_depth, kernel_size, kernel_size)

        #Kernels and biases first set to be random. Needs to take the arguments rather than a tuple.
        self.kernels = np.random.randn(*self.kernels_shape)
        self.biases = np.random.randn(*self.output_shape)



    def forward(self, input):

        # Input is an numpy Array made from the images
        self.input = input
        self.output = np.copy(self.biases)

        for i in range(self.depth):

            for j in range(self.input_depth):

                # Output of the layer is the biases + the input cross correlated with the kernels
                self.output[i] += signal.correlate2d(self.input[j], self.kernels[i, j], "valid")
        
        return self.output
    


    def backwords(self, output_gradient, learning_rate):

        #Initialising the matrices
        kernels_gradient = np.zeros(self.kernels_shape)
        input_gradient = np.zeros(self.input_shape)

        for i in range(self.depth):
            
            for j in range(self.input_depth):
                
                # The kernel error is the input cross correlated with the error given the output
                kernels_gradient[i, j] = signal.correlate2d(self.input[j], output_gradient[i], "valid")

                # The change to the input gradient the error given the output fully convolved with the kernels
                input_gradient[j] += signal.convolve2d(output_gradient, self.kernels[i, j], "full")

        # Updating kernels and biases
        self.kernels -= learning_rate * kernels_gradient
        self.biases -= learning_rate * output_gradient

        return input_gradient

