import numpy as np
from Layer import Layer

class PoolingLayer(Layer):

    def __init__(self, input_shape, filter_size):

        self.input_shape = input_shape
        self.filter_size = filter_size
        input_width, input_height, input_depth = input_shape
        self.input_depth = input_depth
        self.filter_shape = (filter_size, filter_size)
        self.stride = filter_size
        self.numb_windows = input_width/ filter_size
        self.output_shape = (int(input_width/filter_size), int(input_height/filter_size),int(input_depth))

    def forward(self, input):

        self.input = input
        output = np.zeros(self.output_shape)


        for i in range(self.input_depth/self.stride):

            
            
            for j in range(self.filter_size):

                for k in range(self.filter_size): 

                    window = input[(a*j):((a*j) + self.filter_size) , (a*k) : ((a*k) + self.filter_size), i]
                    output[k,j,i] = window.max()  
    





        


 


        return
