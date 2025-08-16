import numpy as np
from Layer import Layer

# TO-DO: Pretty much the same as 1x1 convolution could maybe remove

class FCLayer(Layer):

    def __init__(self, output_size: int):

        self.biases = np.random.rand(output_size,1)
        self.output_size = output_size
        self.weights = np.array([])

        super().__init__()

    def forward(self, input: np.ndarray):

        self.input_shape = input.shape
        self.input = input.reshape((input.size,1))
       # print(f"input size {input.size}")
        #print(f"Full input: {input}")
        #print(f" Input shape: {input.shape}")
        

        if len(self.weights) == 0:

            self.weights = np.random.rand(self.output_size, input.size)

        # No need to flatten as technically done by the final conolutional layer

        out = np.dot(self.weights,self.input)
        
        self.output = out + self.biases
        #print(f"dot: {self.output.size}")
    


        #print(self.output.shape)
        # print(f"Weights: {self.weights[0]}")
        # print(f"Biases: {self.biases}")
        # print(f"Output: {self.output}")

        #self.output = np.dot(self.weights, self.input) + self.biases
        #print(f" Full output: {self.output}")

        return self.output
    
    def backward(self,error_grad: np.ndarray, learning_rate: float):

        weights_gradient = np.dot(error_grad, self.input.T)
        
        input_gradient = np.array(np.dot(self.weights.T, error_grad)) 
        input_gradient = input_gradient.reshape(self.input_shape)

        print(input_gradient.shape)


        self.weights -= weights_gradient * learning_rate
        self.biases -= error_grad * learning_rate

        return input_gradient




        







        
