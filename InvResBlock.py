import numpy as np
from Layer import Layer
from ConvolutionalLayer import ConvolutionLayer
from Activations import RELU6



class convR6():

    def __init__(self, input_channels, output_channels,kernel_size,stride, expansion_ratio):
         

         expanded_input = input_channels * expansion_ratio

         layers = []

        # 3 Phases

        # Expansion phase 
        # Firat layer should be a 1 by 1 point wise convolution of input depth only if the expanse ratio is not 1
        # With RELU6 activation

         if expansion_ratio != 1 and input_channels == output_channels:
            
            layers.extend([ConvolutionLayer(input_channels, kernel_size = 1, depth =  expanded_input),
                          RELU6()
                          ])


        # Depthwise Convolution phase
        # Second Layer should be a 3x3 depth-wise convolution of 1 depth
        # With RELU6 activation

        # Projection Phase
        # Third layer should be a 1x1 Convolution

        pass


class InvResBlock(Layer):

    def __init__(self, input_channels, output_channels, expansion_ratio):



        self.output_channels = output_channels


    

    def forward(self,input):

        pass

    def backward(self,output_gradient)