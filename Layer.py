# Basic functionality of all classes. All Layers from the networks will override and inherit from the class.
import numpy as np 

class Layer:

    def __init__(self):

        self.input = None
   
    def forward(self, input: np.ndarray) -> np.ndarray:

        return input

    def backward(self, error_grad: np.ndarray, learning_rate: float) -> np.ndarray:
        
        return error_grad