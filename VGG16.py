
import numpy as np
import cv2
import os
from Layer import Layer
from ConvolutionalLayer import ConvolutionLayer
from Pooling import MaxPooling
from FullyConnected import FCLayer
from Softmax import Softmax
from Learning import train
from Losses import logLoss,logLossPrime
from DataLoader import DataLoader
from Activations import RELU6
from BatchNormalisation import BatchNorm


class VGG16():

    def __init__(self):

        self.network = [

            ConvolutionLayer(3,1,3),
            RELU6(),
            ConvolutionLayer(1,2,3),
            RELU6(),
            MaxPooling((2,2),2),
            ConvolutionLayer(2,2,3),
            RELU6(),
            ConvolutionLayer(2,2,3),
            RELU6(),
            MaxPooling((2,2),2),
            ConvolutionLayer(2,2,3),
            RELU6(),
            ConvolutionLayer(2,2,3),
            RELU6(),
            ConvolutionLayer(2,2,3),
            RELU6(),
            ConvolutionLayer(2,2,3),
            RELU6(),
            MaxPooling((2,2),2),
            ConvolutionLayer(2,2,3),
            RELU6(),
            ConvolutionLayer(2,2,3),
            RELU6(),
            ConvolutionLayer(2,2,3),
            RELU6(),
            ConvolutionLayer(2,2,3),
            RELU6(),
            MaxPooling((2,2),2),
            ConvolutionLayer(2,2,3),
            RELU6(),
            ConvolutionLayer(2,2,3),
            RELU6(),
            ConvolutionLayer(2,2,3),
            RELU6(),
            ConvolutionLayer(2,2,3),
            RELU6(),
            MaxPooling((2,2),2),
            FCLayer(4096),
            BatchNorm(),
            RELU6(),
            FCLayer(4096),
            BatchNorm(),
            RELU6(),
            FCLayer(8),
            Softmax()
        ]

vgg16 = VGG16()

network = [

    ConvolutionLayer(3,2,3),
    RELU6(),
    ConvolutionLayer(2,2,3),
    RELU6(),
    MaxPooling((2,2),2),
    FCLayer(4096),
    BatchNorm(),
    RELU6(),
    FCLayer(8),
    Softmax()

]

img_paths = []
exp_paths = []

image_path = "C:/Users/james/Documents/Masters/Dissertation/Preprocessed_Train/Train_images"
train_exp_path = "C:/Users/james/Documents/Masters/Dissertation/Preprocessed_Train/Train_expected"
test_image_paths = ""
test_expected_paths =""

train_data = DataLoader(image_path, train_exp_path,3,targetSize=(32,32))

#x_test, y_test = DataLoader([*test_image_paths],[*test_expected_paths],50)




train(network, logLoss, logLossPrime, train_data)





    
        














    


