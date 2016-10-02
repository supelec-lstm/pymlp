from src.Neuron import *
import numpy as nump

class SigmoidNeuron(Neuron):

    def activation_function(self,x):
        """Activation function for the neuron"""
        return 1/(1+nump.exp(x))

    def gradient_activation_function(self,x):
        """Calculate the derivative of the function"""
        return self.activation_function(x)*(1-self.activation_function(x))


