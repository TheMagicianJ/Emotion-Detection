import numpy as np
from Layer import Layer
import math

class BatchNorm(Layer):

    def __init__(self):

        super().__init__()

    def forward(self, input: np.ndarray):

        self.input = input
        print(f"Batch input: {self.input[0:3]}")

        mean = float(np.mean(self.input))
        #print(f"Mean: {mean}")
    
        var = float(np.var(self.input))

        self.output = (self.input - mean)/np.sqrt((var**2) + (1*10**-5))

        #print(f"Batch output{self.output[0:3]}")

        return self.output

        











   