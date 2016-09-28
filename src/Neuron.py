import numpy as np

class Neuron:

    def __init__(self,name,parents,init_function):
        """Constructor used for initialising the neuron, declaring its name, its neuron parents and its weights"""
        self.name = name
        self.parents = parents
        self.w = {}
        self.acc_dJdw = {}
        self.init_function = init_function
        for poids in range(len(parents)):
            self.w[self.parents.name]= init_function()
            self.acc_dJdw[self.parents.name] = 0
        self.children, self.x, self.y, self.dJdx = None
        self.x = np.empty(len(parents))

    def evaluate(self):
        """Return the output of the neuron as a double"""
        if(self.x is None):
            for neuron in self.parents:
                self.x[neuron.name] = neuron.evaluate(self)
        self.y = self.activation_function(np.dot(self.x,self.w))
        return self.y


    def get_gradient(self):
        """Return the gradient of the neuron, based on its inputs and the associated weights"""
        dJdy = 0
        for child in self.children:
            dJdy += child.get_gradient()[self.name]
        dJdy *= self.gradient_activation_function(np.dot(self.x,self.w))
        dhdx = self.w.copy()
        dhdw = self.x.copy()

        dhdx.update((x, y*dJdy) for x, y in dhdx.items() )
        dhdw.update((x, y*dJdy) for x, y in dhdw.items() )

        for child in self.children:
            self.acc_dJdw[child.name] += dhdw.get(child.name)

    def descend_gradient(self,learning_rate,batch_size):
        """Updates the weights related to the neuron"""
        self.w = self.w - learning_rate*self.acc_dJdw/batch_size

    def add_child(self, child):
        """Add a child to the neuron"""
        self.children[child.name] = child

    def reset_memoization(self):
        """Reset memoization"""
        self.x = None
        self.y = None
        self.dJdx = None

    def reset_accumulator(self,x):
        """Reset accumulator"""
        self.acc_dJdw = 0

    def activation_function(self):
        """Activation function for the neuron"""
        raise NotImplementedError(self)

    def gradient_activation_function(self,x):
        """Calculate the derivative of the function"""
        raise NotImplementedError(self)