import numpy as np
from Layer import Layer

class SqueezeExciteLayer(Layer):

    def __init__(self):

        # Certain channels may be more important (contribute more than the rest) than others.
        # We can use gloabl average pooling to find the a coefficient for the channels .
        # And then use a fully connected layer to find out which channel is more important than the others.
        # This way the model may pay more attention to the more performative channels in learning

        network = [

            
        ]

