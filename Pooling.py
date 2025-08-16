import numpy as np
from Layer import Layer
from Conv import pad3D



class MaxPooling(Layer):

    def __init__(self, window_shape: tuple, stride: int, padding: int = 0 ) :

        self.window = window_shape
        self.stride = stride
        
        if padding < 0:

            print("Padding be more than or equal to 0")
            raise SystemExit()
        
        self.padding = padding


    def forward(self,input: np.ndarray):

        
        self.input = input
        
        if self.padding > 0 :

            self.input = pad3D(self.input, self.padding)

        self.input_depth, self.input_width, self.input_height = self.input.shape

        
        if self.window[0] > self.input_width or self.window[1] > self.input_height:

            self.window = (self.input_width, self.input_height)

        # ((input size - pooling window size) / stride ) + 1 Assuming no padding
        self.output_width = ((self.input_width - self.window[0]) // self.stride) + 1
        self.output_height = ((self.input_height - self.window[1]) // self.stride) + 1

        self.output = np.zeros((self.input_depth, self.output_width, self.output_height))
        
        for d in range(self.input_depth):

            for w in range(self.output_width):

                 for h in range(self.output_height):
                    
                    # Finding the max in an n x m sized window at depth d
                    
                    self.output[d,w,h] = input[d,w*self.stride:w*self.stride + self.window[0], h*self.stride: h*self.stride + self.window[1]].max()

        #print(f"Max Pooling: {self.output}")
                    
        return self.output
    
    
    def backward(self, error_grad: np.ndarray, learning_rate: float):

        # Gradient only matter only on the positions where the max was found everything else is left the same so pad out with zero.
        #Initialise an array of zeroes similar to the input
        print(f"pool input shape {self.input.shape}")
        backprop = np.zeros(self.input.shape)

        print(f"pool error {error_grad.shape}")

        for h in range(self.output_height):

            for w in range(self.output_width):
                    
                # Finding the max in an n x m sized window at depth d
                # Adds the error gradient to the array only where the max is found. Boolean at the end determines whether its is added or not as the max is either found there (1) or not (0).
                    
                backprop[:,w*self.stride:w*self.stride + self.window[0], h*self.stride: h*self.stride + self.window[1]] += error_grad[:,w:w+1 ,h:h+1] * (self.output[:,w:w+1,h:h+1] == self.input[:,w*self.stride:w*self.stride + self.window[0], h*self.stride: h*self.stride + self.window[1]])
        
        return backprop[:,self.padding : self.input_width - self.padding, self.padding : self.input_height - self.padding]
    
    
class AveragePooling(Layer):


    def __init__(self, input_shape: tuple, window: tuple, stride: int):

        self.input_shape = input_shape
        self.window= window
        self.stride = stride
        self.output_shape = (input_shape[0], ((self.input_shape[1] - self.window[1]) / self.stride) + 1, ((self.input_shape[2] - self.window[2]) / self.stride) + 1)
        self.output = np.zeros(*self.output_shape)


    def forward(self, input: np.ndarray):

        self.input = input

        for d in range(self.output_shape[0]):

            for h in range(self.output_shape[2]):

                for w in range(self.output_shape[1]):

                   self.output[d,w,h] = np.average(self.input[d,w*self.stride:w*self.stride + self.window[1], h*self.stride: h*self.stride + self.window[2]])

        return self.output
    
    
    def backward(self, error_grad: np.ndarray, learning_rate: float):

        backprop = np.zeros(*self.input_shape)

        for d in range(self.output_shape[0]):

            for h in range(self.output_shape[2]):

                for w in range(self.output_shape[1]):

                    # Because we averaged over the inputs in the window the error is equally spread across the inputs to the pooling window
                    backprop[d, w*self.stride : w*self.stride + self.window[1], h*self.stride : h*self.stride + self.window[2]] += (error_grad[d,h,w] / (self.window[1]* self.window[2])) 

        return backprop
    

class GlobalAveragePooling(Layer):

    def __init__(self):

        super().__init__()


    def forward(self, input: np.ndarray) -> np.ndarray:

        self.input = input
        self.input_shape = input.shape
        self.inp_depth, self.inp_width, self.inp_height = self.input_shape

        self.output = np.zeros((self.inp_depth,1,1))

        for i in range(self.inp_depth):
            
            self.output[i] = np.mean(input)

        return self.output
    
    
    def backward(self, error_grad: np.ndarray, learning_rate: float) -> np.ndarray:

        inp_grads = self.input

        for i in range (self.inp_depth):

             inp_grads += error_grad[i]/ (self.inp_width * self.inp_height)


        return inp_grads

