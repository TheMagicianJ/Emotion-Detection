import numpy as np
from Layer import Layer
from ConvolutionalLayer import ConvolutionLayer
from Activations import RELU6


class InvResBlock(Layer):

    def __init__(self, input_channels, output_channels, expansion_ratio):

        expanded_input = input_channels * expansion_ratio

        layers = []


        # 3 Phases

        # Expansion phase 
        # Firat layer should be a 1 by 1 point wise convolution of input depth only if the expanse ratio is not 1
        # With RELU6 activation

        if expansion_ratio != 1:
            
            layers.extend([ConvolutionLayer(input_channels, kernel_size = 1, depth =  expanded_input),
                          RELU6()
                          ])
        
        layers.extend(
            # Depthwise Convolution phase
            # Second Layer should be a 3x3 depth-wise convolution of 1 depth
            # With RELU6 activation
            [ConvolutionLayer(input_channels, kernel_size = 3, depth = 1),
            RELU6(),

            # Projection Phase
            # Third layer should be a 1x1 Convolution
            ConvolutionLayer(input_channels, kernel_size = 1, depth = output_channels)]
            
            )
        
        self.layers = layers

        

    def forward(self,input):

        output = input

        for layer in self.layers:
           
           output = layer.forward(output)

        return output

    def backward(self,output_grad,learning_rate):

        pass