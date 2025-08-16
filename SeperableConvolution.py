import numpy as np
from Layer import Layer
from scipy import signal
from ConvolutionalLayer import ConvolutionLayer
from Conv import crossCorrelation2D,convolution2D



class DWConvolution(ConvolutionLayer):

    def __init__(self,input_channels: int, kernel_size: int, stride: int):

        super().__init__(input_channels, kernel_size, stride, padding = kernel_size//2)


        self.kernels_shape = (input_channels,kernel_size,kernel_size)
        self.kernels = np.random.randn(*self.kernels_shape)
        self.input_channels = input_channels
   


    def forward(self, input: np.ndarray):

        self.input = input

        if self.padding > 0:

            padded = []

            for i in range(input.shape[0]):

                padded.append(np.pad(input[0], (self.padding,self.padding)))

            self.input = np.array(padded)
        
        self.input_shape = self.input.shape
        self.input_depth, self.input_width, self.input_height = self.input.shape
        

        if self.kernel_size > self.input_height or self.kernel_size > self.input_width:

            self.kernels = np.random.randn(self.input_channels, self.input_width, self.input_height)
            self.kernels_shape = self.kernels.shape
            self.kernel_size = self.input_width

        output_width = ((self.input_width - self.kernel_size)//self.stride) + 1
        output_height = ((self.input_height - self.kernel_size)//self.stride) + 1


        self.output_shape = (self.input_depth, output_width, output_height)

        if len(self.biases) == 0:

            self.biases = np.random.rand(*self.output_shape)

        self.output = np.copy(self.biases)

        for c in range(self.input_channels): 

                # Output of the layer is the biases + the input cross correlated with the kernel of the same depth
                self.output[c] += crossCorrelation2D(self.input[c], self.kernels[c], self.stride)

        print(f"Depth-wise Output: {self.output.shape}")
        
        return self.output
    

    def backward(self, error_grad, learning_rate):

        #Initialising the matrices
        kernels_gradient = np.zeros(self.kernels_shape)
        input_gradient = np.zeros(self.input_shape)

        print(f"DW Info - Input Shape: {self.input_shape} Kernel shape: {self.kernels.shape}, Error shape: {error_grad.shape}, Stride: {self.stride}")

        for i in range(self.input_depth):

            # The kernel error is the input cross correlated with the error given the output
            kernels_gradient[i] += crossCorrelation2D(self.input[i], error_grad[i], self.stride, output_size= self.kernel_size)
            
            # The change to the input gradient the error given the output fully convolved with the kernels
            input_gradient[i] += convolution2D(error_grad[i],self.kernels[i], self.stride, "full", output_size = self.input_width)

        # Updating kernels and biases
        self.kernels -= kernels_gradient * learning_rate
        self.biases -= error_grad * learning_rate
        
        print("DW Backprop Done")

        return input_gradient[:, self.padding : self.input_width - self.padding, self.padding : self.input_height - self.padding]
    
# class PWConvolution(ConvolutionLayer):

#     def __init__(self,input_channels, depth, kernel_size,stride):

#         #Point wise convolution has a kernel size of 1x1 with a depth of n
#         kernel_size = 1

#         super().__init__(input_channels, depth, kernel_size, stride)

        

        







            








    






  