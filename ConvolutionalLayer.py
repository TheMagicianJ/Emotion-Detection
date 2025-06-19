import numpy as np
from Layer import Layer
from scipy import signal
import torch
import torch.nn.functional as F

class ConvolutionLayer(Layer):



    def __init__(self,input_channels,  depth, kernel_size, stride):

        # kernel_Size represents the dimensions of each matrix in a kernel
        # depth represents how many kernels

        self.depth = depth
        self.stride = stride
        self.input_channels = input_channels
        self.kernel_size = kernel_size
  
        # The depth of the kernels 
        self.kernels_shape = (depth, input_channels, kernel_size, kernel_size)  

        #Kernels and biases first set to be random. Needs to take the arguments rather than a tuple.
        self.kernels = np.random.randn(*self.kernels_shape)

        self.biases = None


    def forward(self, input):

        # Input is an numpy Array made from the images
        self.input = input
        self.input_shape = input.shape
        input_depth, input_width, input_height = self.input.shape
        # The amount of output matrices should be the same as number of kernels
        self.output_shape = (self.depth, (((input_height - self.kernel_size)//self.stride) + 1), (((input_width - self.kernel_size)//self.stride) + 1))

        if self.biases == None:

            self.biases = np.random.randn(*self.output_shape)

        
        self.output = np.copy(self.biases)

        # Turn to Tensor (TO BE CHANGED)
        t_input = torch.from_numpy(self.input)
        t_kernels = torch.from_numpy(self.kernels)

        for i in range(self.depth):

            for j in range(self.input_channels): 

                # Output of the layer is the biases + the input cross correlated with the kernels
                
                self.output[i] += F.conv2d(t_input[j], t_kernels[i, j], stride = self.stride).numpy()

        return self.output
    


    def backwords(self, output_gradient, learning_rate):

        #Initialising the matrices
        kernels_gradient = np.zeros(self.kernels_shape)
        input_gradient = np.zeros(self.input_shape)

        for i in range(self.depth):
            
            for j in range(self.input_channels):
                
                # The kernel error is the input cross correlated with the error given the output
                kernels_gradient[i, j] = signal.correlate2d(self.input[j], output_gradient[i], "valid")

                # The change to the input gradient the error given the output fully convolved with the kernels
                input_gradient[j] += signal.convolve2d(output_gradient, self.kernels[i, j], "full")

        # Updating kernels and biases
        self.kernels -= learning_rate * kernels_gradient
        self.biases -= learning_rate * output_gradient

        return input_gradient


