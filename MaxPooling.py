import numpy as np
from Layer import Layer


class MaxPooling(Layer):

    def __init__(self, input_shape, window_shape, stride, ) :

        self.window = window_shape
        self.stride = stride

        #Need to store the position of the max for bakpragation
        self.max_positions = ()
        self.input_shape = input_shape
    


    def forward(self,input):

        # ((input size - pooling window size) / stride ) + 1 Assuming no padding
        #self.input = input
        output_width = ((self.input_shape[0] - self.window[0]) / self.stride) + 1
        output_hieght =((self.input_shape[1] - self.window[1]) / self.stride) + 1
        self.output = np.zero(output_width, output_hieght, self.input_shape[2])

        for d in range(self.input_shape[2]):

            for w in range(output_width):

                 for h in range(output_hieght):
                    
                    # Finding the max in an n x m sized window at depth d
                    
                    self.output[w,h,d] = input[w*self.window[0]:w*self.window[0] + self.window[0], h*self.window[1]: h*self.window[1] + self.window[1],d] .max()

                    self.max_positions = np.where(input == self.output[w,h,d])

        return self.output
    
    
    def backward(self, output_grad):

        # Gradient only matter only on the positions where the max was found everything else is left the same so pad out with zero.
        #Initialise an array of zeroes similar to the input
        backprop = np.zeros(self.input_shape)

        for d in range(self.input_shape[2]):

            for w in range(self.input_shape[0]):

                 for h in range(self.input_shape[1]):
                    
                    # Finding the max in an n x m sized window at depth d
                    # Adds the error gradient to the array only where the max is found. Boolean at the end determines whether its is added or not as the max is either found there (1) or not (0).
                    
                    backprop[w*self.window[0]:w*self.window[0] + self.window[0], h*self.window[1]: h*self.window[1] + self.window[1],d] = output_grad[w:w+1,h:h+1,d] * (self.output[w:w+1,h:h+1,d] == self.input[w*self.window[0]:w*self.window[0] + self.window[0], h*self.window[1]: h*self.window[1] + self.window[1],d] .max())

            return backprop

        

