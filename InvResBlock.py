import numpy as np
from Layer import Layer
from ConvolutionalLayer import ConvolutionLayer
from SeperableConvolution import DWConvolution
from Activations import RELU6


class InvResBlock(Layer):

    def __init__(self, input_channels, output_channels, expansion_ratio, stride):

        expanded_input = input_channels * expansion_ratio

        layers = []
        # If the number of input channels and the number of output channels are the same then we add the input to the output.
        self.res = input_channels == output_channels and stride == 1


        # 3 Phases

        # Expansion phase(Pointwise Convolution)
        # Firat layer should be a 1 by 1 point wise convolution of input depth only if the expanse ratio is not 1
        # With RELU6 activation

        if expansion_ratio != 1:
            
            layers.extend([ConvolutionLayer(input_channels, kernel_size = 1, depth = expanded_input, stride = 1),
                          RELU6()
                          ])
        
        layers.extend(
            # Depthwise Convolution phase
            # Second Layer should be a 3x3 depth-wise convolution of 1 depth
            # With RELU6 activation
            [DWConvolution(expanded_input, kernel_size = 3, depth = expanded_input, stride = stride),
            RELU6(),

            # Projection Phase (Pointwise Convolution)
            # Third layer should be a 1x1 Convolution of n depth
            ConvolutionLayer(expanded_input, kernel_size = 1, depth = output_channels, stride = 1)]
            
            )
        
        self.layers = layers



    def forward(self,input):

        output = input

        for layer in self.layers:
           
           output = layer.forward(output)

        if self.res:

            output += input

        return output

    def backward(self,output_grad,learning_rate):

        pass