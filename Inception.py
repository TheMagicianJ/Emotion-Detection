import numpy as np
from Layer import Layer


class InceptionLayer(Layer):

    def __init__(self, branches: list):

        self.branches = branches
        self.num_branches = len(branches)
        self.partitions = []

    def forward(self, input: np.ndarray) -> np.ndarray:

        inpt = input
        self.input = input
        self.input_shape = input.shape
        print(f"Inception Input{self.input.shape}")

        self.outputs = []
        self.output_depths = [] 

        for i in range(self.num_branches):

            output = inpt

            for layer in self.branches[i]:

                output = layer.forward(output)

            
            self.outputs.append(output) 
            self.output_depths.append(output.shape[0])

        self.output = np.concatenate(self.outputs)

        return self.output
    

    def backward(self, error_grad: np.ndarray, learning_rate: float) -> np.ndarray:

        if len(self.partitions) == 0 :
         
            index = 0

            for i in range(len(self.output_depths) - 1):
                
                index += self.output_depths[i]
                self.partitions.append(index)

        errors = np.split(error_grad, self.partitions, axis = 0)

        # error_list = []
        # for error in errors:

        #     error_list.append(error)

        # error_list.reverse()

        output = np.zeros(self.input.shape)
        input_list = []

        # for error in errors:

        #     error_list.append(error)
        
        # error_list = error_list.reverse()


        print(f"Error 1 sshape: {errors[1].shape}")

        print(self.partitions)

        for i in range(self.num_branches):

            error = errors[i]

            self.branches[i].reverse()

            for layer in self.branches[i]:

                error = layer.backward(error, learning_rate)

            output += error

            self.branches[i].reverse()

            
        return output

