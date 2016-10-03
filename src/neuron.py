import random
from math import exp
from math import tanh

class Neuron ():

    def __init__(self, name, parents):
        self.name = name
        self.parents = parents
        self.children = []
        self.w = []
        self.x = []
        self.y = 0
        self.dJdx = {}
        self.acc_dJdw = 0

        for i in range (len(parents)):
            parents[i].add_children(self)

        for i in range (len(parents)):
            self.w +=[random.randint(-100, 100) * 0.01]


    def add_children (self,neurone):
        self.children+=[neurone]

    def evaluate (self):

        for i in range (len(self.parents)):
            self.x+=self.parents[i].evaluate()
            self.y+=self.w[i]*self.x[i]

        return self.activation_function(self.y)

    def activation_function (self, u):
        return 0

    def gradient_activation_function (self,z):
        return 1

    """def gradient_activaion_function(self,u):
        return 0"""

    def reset_memorization(self):
        self.x=[]
        self.y=0
        self.dJdx={}

    def reset_accumulator (self):
        self.acc_dJdw=0

    def get_gradient(self):

        dJdh=0

        for child in self.children :
            child_gradient=child.get_gradient()
            dJdh+=child_gradient[self.name]

        self.dJdx={}

        for i,parent in self.parents :
            self.dJdx={parent.name: -dJdh*self.w[i]}

        self.acc_dJdw+= dJdh*self.x

        return self.dJdx


    def descend_gradient(self, learning_rate, batch_size):
        self.w=self.w-learning_rate/batch_size*self.acc_dJdw
        self.reset_accumulator()



class InputNeuron (Neuron):

    def __init__(self, name, value):
        self.name=name
        self.y=0
        self.set_value(value)

    def set_value (self, value):
        self.y=value

    def evaluate (self):
        return self.y

    def activation_function(self, z):
        return self.y


class LinearNeuron (Neuron):

    def activation_function(self, z):
        return z

    def gradient_activation_function (self, z):
        return 1

class Sigmoid_Neuron (Neuron):

    def activation_function(self, z):
        return (1/(1+exp(-z)))

    def gradient_activation_function (self,z):
        return

class Tanh_Neuron (Neuron):

    def activation_function(self, z):
        return (tanh(z))

    def gradient_activation_function (self,z):
        return 1-tanh(z)**2

class ReluNeuron (Neuron):

    def activation_function(self, z):
        if (z>0):
            return z
        else :
            return 0

    def gradient_activation_function (self,z):
        if (z>0):
            return 1
        else :
            return 0

class CostNeuron(Neuron):

    def __init(self, name, expected_outputs, outputs):
        self.name=name
        self.expected_outputs=expected_outputs
        self.outputs=outputs

    def evaluate(self):
        y =[]

        for i in  range (len(self.outputs)):
            y.append((1/2)*(self.expected_outputs[i]-self.outputs[i])**2)

        return y

    def get_gradient(self):
        gradient =0
        for i in range (len(self.expected_outputs)):
            gradient+=self.expected_outputs[i]-self.outputs[i]
        return gradient




