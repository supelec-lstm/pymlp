import numpy as np
from math import *

class Neuron:
    
    '''create an independant neuron'''

    def __init__(self, name, parents = [], init_function = None):

        '''initialization of the neuron'''

        self.name = name
        
        self.parents = parents
        self.children = []

        self.w = init_function(len(parents)) if init_function else np.array([np.random.normal(0, 0.1) for _ in range(len(parents))])
        

        for parent in parents:
            parent.add_child(self)

        #memoization variables

        self.dJdx = {}
        self.x = None
        self.y = None

        #gradient accumulator for batches

        self.acc_dJdw = None


    def evaluate(self):

        ''' compute the output of the neuron'''
        if not self.y:
            self.x = np.array([parent.evaluate() for parent in self.parents])
            print(self.name)
            print(len(self.parents))
            print(self.x)
            print(self.w)
            self.y = self.activation_function(self.x.T.dot(self.w))

        return self.y


    def get_gradient(self):
        '''return the gradient of the error in respect to the input, memorize it in dJdx 
        and add the gradient in respect to te weight to the accumulator'''

        if not self.dJdx:

            dJdh = 0

            for child in self.children:
                g = child.get_gradient()
                dJdh += g[self.name]

            dJdh *= self.gradient_activation_function(self.x.T.dot(self.w))

            self.dJdx = {}

            for i, parent in enumerate(self.parents):
                self.dJdx[parent.name] = dJdh*self.w[i]              

            self.acc_dJdw += dJdh*self.x    #le x sera t il la?? => il faut evaluer avant d'entrainer

        return self.dJdx


    def descend_gradient(self, learning_rate = 0.3, batch_size = 1):

        '''apply changes to the weights'''

        self.w -= learning_rate/batch_size*self.acc_dJdw
        self.reset_accumulator()


    def add_child(self, child):

        '''add the newly created child to the child list'''
        
        self.children.append(child)


    def reset_memoization(self):

        '''reset all the memoization variables : x, y, and dJdw'''

        self.x = None
        self.y = None
        self.dJdw = {}


    def reset_accumulator(self):

        '''reset the gradient accumuator'''

        self.acc_dJdw = None


    def activation_function(self, t):
        
        return None


    def gradient_activation_function(self, v):

        return None











class InputNeuron(Neuron):
    '''create an input Neuron for the network'''

    def __init__(self, name, parents = [], init_function = None):
        '''initialize the input neuron with its value'''

        Neuron.__init__(self, name, parents, init_function)

    def set_value(self, value):
        '''change the value of the input neuron'''

        self.value = value

    def get_gradient(self):

        for parent in self.parents:
            parent.get_gradient()


    def activation_function(self, x):
        '''override the activation function to give as output the input value of the network'''

        return self.value

    def evaluate(self):

        return self.value


class LinearNeuron(Neuron):
    '''create a Neuron with a linear activation function'''

    def __init__(self, name, parents = [], init_function = None):
        '''initialize a linear neuron'''

        Neuron.__init__(self, name, parents, init_function)


    def activation_function(self, x):
        '''linear activation function'''

        return x


    def gradient_activation_function(self, x):
        '''gradient of the activation function'''

        return 1


class SigmoidNeuron(Neuron):
    '''create a Neuron with a sigmoid activation function'''

    def __init__(self, name, parents = [], init_function = None):
        '''initialize a sigmoid neuron'''

        Neuron.__init__(self, name, parents, init_function)


    def activation_function(self, x):
        '''sigmoid activation function'''

        return 1/(1+exp(-x))


    def gradient_activation_function(self, x):
        '''gradient of the activation function'''

        return self.activation_function(x)*(1-self.activation_function(x))


class TanhNeuron(Neuron):
    '''create a Neuron with a hyperbolic tangent activation function'''

    def __init__(self, name, parents = [], init_function = None):
        '''initialize a hyperbolic tangent neuron'''

        Neuron.__init__(self, name, parents, init_function)


    def activation_function(self, x):
        '''hyperbolic tangent activation function'''

        return tanh(x)


    def gradient_activation_function(self, x):
        '''gradient of the activation function'''

        return 1-tanh(x)**2



class ReluNeuron(Neuron):
    '''create a Neuron with a Relu activation function'''

    def __init__(self, name, parents = [], init_function = None):
        '''initialize a Relu neuron'''

        Neuron.__init__(self, name, parents, init_function)


    def activation_function(self, x):
        '''Relu activation function'''

        return x if x >= 0.0 else 0


    def gradient_activation_function(self, x):
        '''gradient of the activation function'''

        return 1 if x > 0.0 else 0



class SoftMaxNeuron(Neuron):
    '''create a Neuron with a softmax activation function'''

    def __init__(self, name, parents = [], init_function = None):
        '''initialize a softmax neuron'''

        Neuron.__init__(self, name, parents, init_function)


    def activation_function(self, x):
        '''softmax activation function'''

        pass


    def gradient_activation_function(self, x):
        '''gradient of the activation function'''

        return activation_function(x)(1-activation_function(x))



class SquaredErrorNeuron(Neuron):
    '''create a neuron to compute the suarred error cost function'''

    def __init__(self, name, parents = [], init_function = None):
        '''initialize a squrred error neuron'''

        Neuron.__init__(self, name, parents, init_function)

        #fixed place for the cost neuron giving fixed weights: 
        #the first parent is the expected output of the network and the second is the calculated output

        self.w = np.array([-1,1])

    def activation_function(self, x):
        '''squarred error activation function'''

        return x**2

    def gradient_activation_function(self, x):
        return 1


    def get_gradient(self):
        '''return the derivative of the error in respect to the output'''

        g={}
        for parent in self.parents:
            g[parent.name] = self.x.T.dot(self.w)

        return g


    def descend_gradient(self):
        
        self.w=self.w



#class CrossEntropyNeuron(Neuron):

