import numpy as np
import cv2
import os
from Layer import Layer


class EffecientLayer(Layer):

    def __init__(self,input):

        self.input = input

        invResSettings = [

        # Inverse Residual Block needs the parameters (t,c,n,s)
        # t is the exapnsion factor (i.e. how much we expand the input dimensions by)
        # c is the channel depth
        # N is the number of convolution layers/blocks within the block
        # s is the stride. Stride will be 1

          # t, c, n, s


        ]
        

    def forward(self, input):

        pass

    def backword(self, output_grad, learning_rate):

        pass

