import numpy as np

class SigmoidNeuron(Neuron):

    def activation_function(self,x):
        """Activation function for the neuron"""
        return 1/(1+np.exp(x))

    def gradient_activation_function(self,x):
        """Calculate the derivative of the function"""
        return activation_function(x)*(1-activation_function(x))


