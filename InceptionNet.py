import numpy as np
from Inception import InceptionLayer
from SeperableConvolution import DWConvolution
from ConvolutionalLayer import ConvolutionLayer
from Pooling import MaxPooling, GlobalAveragePooling
from Softmax import Softmax
from FullyConnected import FCLayer
from Activations import RELU6
from DataLoader import DataLoader
from Learning import train
from Losses import logLoss, logLossPrime

class InceptionNet():

    def __init__(self):
        
        self.network = [

            ConvolutionLayer(16,3,1),
            RELU6(),

            InceptionLayer( branches = [
                
                #Branch 1
                [
                ConvolutionLayer(8,1,1),
                RELU6()
                ],
 
                #Branch 2
                [
                ConvolutionLayer(8,1,1),
                RELU6(),
                ConvolutionLayer(16,3,1, padding = 1),
                RELU6()
                ],
                
                #Branch 3
                [
                ConvolutionLayer(4,1,1),
                RELU6(),
                ConvolutionLayer(8,5,1,padding = 2),
                RELU6(),
                ],

                #Branch 4
                [
                MaxPooling((3,3),1,1),
                ConvolutionLayer(8,1,1),
                RELU6()
                ]
            ]),

            MaxPooling((2,2),2),

            InceptionLayer( branches=[ 

                #Branch 1
                [
                ConvolutionLayer(16,1,1),
                RELU6(),
                ],

                #Branch 2
                [
                ConvolutionLayer(12,1,1),
                RELU6(),
                ConvolutionLayer(16,3,1, padding = 1),
                RELU6(),
                ],

                #Branch 3
                [
                ConvolutionLayer(4,1,1),
                RELU6(),
                ConvolutionLayer(8,5,1, padding = 2),
                RELU6(),
                ],

                #Branch 4
                [
                MaxPooling((3,3),1, padding = 1)
                ]
            ]),

            GlobalAveragePooling(),
            FCLayer(8),
            Softmax()
            
        ]


incept = InceptionNet()

img_paths = []
exp_paths = []

image_path = "C:/Users/james/Documents/Masters/Dissertation/Preprocessed_Train/Train_images"
train_exp_path = "C:/Users/james/Documents/Masters/Dissertation/Preprocessed_Train/Train_expected"
test_image_paths = ""
test_expected_paths =""

train_data = DataLoader(image_path, train_exp_path,3, targetSize = (64,64))

#x_test, y_test = DataLoader([*test_image_paths],[*test_expected_paths],50)




train(incept.network, logLoss, logLossPrime, train_data)



    

