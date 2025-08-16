import numpy as np
from Layer import Layer
from Conv import crossCorrelation3D, convolution3D

class ConvolutionLayer(Layer):

    def __init__(self,  depth: int, kernel_size: int, stride: int = 1, padding: int = 0):

        # kernel_Size represents the dimensions of each matrix in a kernel
        # depth represents how many kernels
        self.depth = depth
        self.stride = stride
        self.padding = padding

        self.kernel_size = kernel_size
        self.kernels = []
        self.biases = []

        if padding < 0 or stride < 0:

            print(" Padding and stride must be greater than or equal to 0")
            raise SystemExit()
        
        


    def forward(self, input: np.ndarray):

        # Input is an numpy Array made from the images
        self.input = input

        print(f"Convolution Input {self.input.shape}")
  

        if self.padding > 0:

            padded = []

            for i in range(input.shape[0]):

                padded.append(np.pad(input[0], (self.padding,self.padding)))

            self.input = np.array(padded)

        self.input_shape = self.input.shape
        self.input_channels, self.input_width, self.input_height = self.input.shape
            
        #(input[0])

        if len(self.kernels) == 0:

            if self.kernel_size > self.input_height or self.kernel_size > self.input_width:

                self.kernels = np.random.randn(self.depth,self.input_channels, self.input_width, self.input_height)
                self.kernels_shape = self.kernels.shape
                self.kernel_size = self.input_width

            else:

                self.kernels = np.random.randn(self.depth, self.input_channels, self.kernel_size, self.kernel_size)
                self.kernels_shape = self.kernels.shape
 

        # The amount of output matrices should be the same as number of kernels
        output_width =((self.input_width - self.kernel_size) // self.stride) + 1
        output_height =  ((self.input_height - self.kernel_size) // self.stride) + 1

        self.output_shape = (self.depth, output_width,output_height)

        if len(self.biases) == 0:

            self.biases = np.random.randn(*self.output_shape)

        self.output = np.copy(self.biases)

        for i in range(self.depth):

            # Output of the layer is the biases + the input cross correlated with the kernels
            
            self.output[i] += crossCorrelation3D(self.input, self.kernels[i], self.stride)
            

        #print(f"Convilutional: {self.output.shape}")

        #print(f"Conv_Output { self.output.shape}")



        print(f"Convolution Finished. Output Shape:{self.output_shape}")

        return self.output


    def backward(self, error_grad: np.ndarray, learning_rate: float):

        #Initialising the matrices
        kernels_gradient = np.zeros(self.kernels_shape)
        input_gradient = np.zeros(self.input_shape)

        # The kernel error is the input cross correlated with the error given the output
        print(f"Input shape: {self.input.shape}")
        print(f"Error grad shape: {error_grad.shape}")
        kernels_gradient = crossCorrelation3D(self.input, error_grad, self.stride)

        for i in range(self.depth):
            
            for j in range(self.input_channels):
                
                # The change to the input gradient the error given the output fully convolved with the kernels

                # print(f"Eroor grad shape: {error_grad.shape}")
                # print(f"Kernels shape {self.kernels_shape}")
                
                input_gradient[j] += convolution3D(error_grad, self.kernels[i],   self.stride, "full")

        # Updating kernels and biases
        self.kernels -= kernels_gradient * learning_rate
        self.biases -= error_grad * learning_rate

        print("Back")

        print(input_gradient.shape)

        #print(f"input_grad: {input_gradient.shape}")

        # End by undoing padding

        return input_gradient[:, self.padding : self.input_width - self.padding, self.padding : self.input_height - self.padding]


