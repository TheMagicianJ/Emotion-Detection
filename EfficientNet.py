import numpy as np
import cv2
import os
from Layer import Layer
from ConvolutionalLayer import ConvolutionLayer
from InvResBlock import InvResBlock
from Pooling import AveragePooling
from BatchNormalisation import BatchNorm
from FullyConnected import FCLayer
from Softmax import Softmax
from Learning import train
from DataLoader import DataLoader
from Losses import logLoss, logLossPrime
from Activations import Swish
from Pooling import GlobalAveragePooling
from InvResBlock import InvResBlock
from BatchNormalisation import BatchNorm


class EffecientNetB0():

    def __init__(self):

        self.network = [
            
            ConvolutionLayer(3,32,3,2),
            BatchNorm(),
            Swish(),

            InvResBlock(32,16,1,1,3),
        
            InvResBlock(16,24,6,2,3),
            InvResBlock(24,24,6,2,3),

            InvResBlock(24,40,6,2,5),
            InvResBlock(40,40,6,2,5),

            InvResBlock(24,80,6,2,3),
            InvResBlock(80,80,6,2,3),
            InvResBlock(80,80,6,2,3),

            InvResBlock(80,112,6,1,5),
            InvResBlock(112,112,6,1,5),
            InvResBlock(112,112,6,1,5),

            InvResBlock(112,192,6,2,5),
            InvResBlock(192,192,6,2,5),
            InvResBlock(192,192,6,2,5),
            InvResBlock(192,192,6,2,5),

            InvResBlock(192,320,6,1,3),

            ConvolutionLayer(320,1280,1,1),
            BatchNorm(),
            Swish(),
            GlobalAveragePooling(),
            FCLayer(8),
            Softmax(),

        ]


b1 = [

    ConvolutionLayer(8,3,2, padding = 1),
    # 1 input_channels - 2 expansion_ratio - 3 output_channels - 4 kernel_size - 5 stride
    InvResBlock(8,1,8,3,1),
    InvResBlock(8,6,16,3,2),
    InvResBlock(16,6,24,5,2),
    ConvolutionLayer(32,1,1),
    GlobalAveragePooling(),
    FCLayer(8),
    BatchNorm(),
    Softmax()

]

b_0 = EffecientNetB0()


img_paths = []
exp_paths = []

image_path = "C:/Users/james/Documents/Masters/Dissertation/Preprocessed_Train/Train_images"
train_exp_path = "C:/Users/james/Documents/Masters/Dissertation/Preprocessed_Train/Train_expected"
test_image_paths = ""
test_expected_paths =""

train_data = DataLoader(image_path, train_exp_path,3, targetSize= (32,32))
#x_test, y_test = DataLoader([*test_image_paths],[*test_expected_paths],50)


train(b1, logLoss, logLossPrime, train_data)


