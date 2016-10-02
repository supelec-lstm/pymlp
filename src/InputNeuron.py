from src.Neuron import *

class InputNeuron(Neuron):

    def __init__(self, name, parents, init_function):
        """Constructor used for initialising the neuron, declaring its name, its neuron parents and its weights"""
        self.name = name
        self.parents = parents
        self.w = {}
        self.acc_dJdw = 0
        self.init_function = init_function
        for parent in self.parents:
            self.w[parent.name] = init_function(1)
            parent.add_child(self)
        self.children, self.x, self.y, self.dJdx = [], {}, {}, {}
        self.value = 0

    def set_value(self,value):
        self.value = value

    def activation_function(self,x):
        """Activation function for the neuron"""
        return self.value

    def gradient_activation_function(self,x):
        """Calculate the derivative of the function"""
        return 0


