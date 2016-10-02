import numpy as np

class Neuron:

    def __init__(self,name,parents,init_function):
        """Constructor used for initialising the neuron, declaring its name, its neuron parents and its weights"""
        self.name = name
        self.parents = parents
        self.w = {}
        self.acc_dJdw = 0
        self.init_function = init_function
        for parent in self.parents:
            self.w[parent.name]= init_function(1)
            parent.add_child(self)
        self.children, self.x, self.y, self.dJdx = [],{},{},{}

    def evaluate(self):
        """Return the output of the neuron as a double"""
        if self.x == {}:
            for neuron in self.parents:
                self.x[neuron.name] = neuron.evaluate()
        if self.x is not None:
            argument = sum(self.x[k]*self.w[k] for k in self.x)
            self.y = self.activation_function(argument)

        else :
            self.y = self.activation_function(0)
        return self.y


    def get_gradient(self):
        """Return the gradient of the neuron, based on its inputs and the associated weights"""
        dJdy = 0
        for child in self.children:
            dJdy += child.get_gradient()[self.name]
        dJdy *= self.gradient_activation_function(sum(self.x[k]*self.w[k] for k in list(self.x.keys())))
        dhdx = self.w.copy()
        dhdw = self.x.copy()

        dhdx.update((x, y*dJdy) for x, y in dhdx.items() )
        dhdw.update((x, y*dJdy) for x, y in dhdw.items() )

        self.acc_dJdw += dhdw.get(self.name)

    def descend_gradient(self,learning_rate,batch_size):
        """Updates the weights related to the neuron"""
        """self.w = self.w - learning_rate*self.acc_dJdw/batch_size"""
        self.w.update((x,w-learning_rate*self.acc_dJdw/batch_size) for x,w in self.w.items())

    def add_child(self, child):
        """Add a child to the neuron"""
        if self.children is not None:
            self.children.append(child)

    def reset_memoization(self):
        """Reset memoization"""
        self.x = None
        self.y = None
        self.dJdx = None

    def reset_accumulator(self):
        """Reset accumulator"""
        self.acc_dJdw = 0

    def activation_function(self,x):
        """Activation function for the neuron"""
        raise NotImplementedError(self)

    def gradient_activation_function(self,x):
        """Calculate the derivative of the function"""
        raise NotImplementedError(self)