
class Layer:

    def __init__(self):

        self.input = None
        self.output = None

    def forward(self, input):

        return input

    def backward(self, output_grad, learning_rate):
        
        return output_grad