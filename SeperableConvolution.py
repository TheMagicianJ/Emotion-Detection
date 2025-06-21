import numpy as np
from Layer import Layer
from scipy import signal
from ConvolutionalLayer import ConvolutionLayer
import torch
import torch.nn.functional as F



class DWConvolution(ConvolutionLayer):

    def __init__(self,input_channels, depth, kernel_size, stride):

        super().__init__(input_channels, depth, kernel_size, stride)

        self.kernels_shape = (input_channels,kernel_size,kernel_size)
        self.kernels = np.random.randn(*self.kernels_shape)

    def forward(self,input):

        self.input = input
        input_depth, input_width, input_height = self.input.shape
        self.output_shape = (self.depth, (((input_height - self.kernel_size)//self.stride) + 1), (((input_width - self.kernel_size)//self.stride) + 1))

        if self.biases == None:

            self.biases = self.output_shape

        self.ouput = np.copy(self.output_shape)

        # Turn to Tensor (TO BE CHANGED)

        t_input = torch.from_numpy(self.input)
        t_kernels = torch.from_numpy(self.kernels)
        
        for c in range(self.input_channels): 

            # Output of the layer is the biases + the input cross correlated with the kernel of the same depth
                
            self.output[c] += F.conv2d(t_input[c], t_kernels[c], stride = self.stride).numpy()

        return self.output
    
# class PWConvolution(ConvolutionLayer):

#     def __init__(self,input_channels, depth, kernel_size,stride):

#         #Point wise convolution has a kernel size of 1x1 with a depth of n
#         kernel_size = 1

#         super().__init__(input_channels, depth, kernel_size, stride)

        

        







            








    






  