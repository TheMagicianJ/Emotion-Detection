import numpy as np
from Activation import Activation

    

class Tanh(Activation):

    def __init__(self):
            
        def tanh(x):
                
             return np.tanh(x)
        
        def inverse_tanh(x):

            return
            
            # Super method to intialise the Activation with the functions
        super().__init__(tanh,inverse_tanh)

        def test():

            return


    # TO-DO: Need to implement the new activations for Inception and Effecicient. Sigmoidal?
    
class Act(Activation):

    def __init__(self):

        def act_forward(x):
                
            return x
            
        def act_backward(x):

            return x
            
        super().__init__(act_forward,act_backward)

# TO-DO implement RELU6 activation

class RELU6(Activation):

    def __init__(self):


        def activation(input):

            # RELU6 returns 0 is the input is less than 0 and clamps the max value to 6
            input[input < 0] = 0
            input[input > 6] = 6

            self.input = input

            return self.input

        def activation_prime(output_gradient):

            # For backprop the function need to check if the input given was below 0. If the inpput is below 0 times the input by zero if anything else multiply by 1  m

            x = self.input
            x[x > 0] = 1

            print(f"RELU6 Input shape : {self.input.shape}")

            #print(np.multiply(output_gradient,x))

            return np.multiply(output_gradient,x)

        super().__init__(activation, activation_prime)

class Swish(Activation):

    def __init__(self):

        def activation(input):

            input *= 1/1+np.exp(-input)

            return input

        def activation_prime(x):

            output = ((x + np.sinh(x)) / (4 * (np.cosh(x)**2))) + 0.5


            return output

        super().__init__(activation, activation_prime)

        
            
        



