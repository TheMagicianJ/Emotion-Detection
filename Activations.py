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

        def forward(x):
            
            pass

        def backward(x):

            pass

        super().__init__(forward, backward)

        
            
        



