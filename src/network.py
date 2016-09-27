from neuron import *
import numpy as np

class Network:
    '''creates an abstract network assembling neurons'''

    def __init__(self, neurons, inputs, outputs, expected_outputs, cost_neuron):
        '''initialize a network of neurons'''

        self.neurons = neurons
        self.inputs = inputs
        self.outputs = outputs
        self.expected_outputs = expected_outputs
        self.cost_neuron = cost_neuron




    def propagate(self, x):
        '''propagate the input in the entire network,
        each neuron memorize its values'''

        for xi, neuron in zip(x, self.inputs):
            neuron.set_value(xi)

        self.reset_memoization()

        return np.array([output.evaluate() for output in self.outputs])



    def back_propagate(self, x, y):
        '''compute the gradient in each neuron of the network 
        considering the desired output y for the input x'''
        
        self.propagate(x)

        for yi, neuron in zip(y, self.expected_outputs):
            neuron.set_value(yi)

        self.cost_neuron.evaluate()

        for neuron in self.inputs:
            neuron.get_gradient()


    def descend_gradient(self, learning_rate = 0.3, batch_size = 1):
        '''apply the gradient descent in each neuron'''

        for neuron in self.neurons:
            neuron.descend_gradient(learning_rate, batch_size)


    def batch_gradient_descent(self, X, Y, learning_rate = 0.3):      #added learning rate argument
        '''realize a gradient descent over an entire batch at the time'''

        for x, y in zip(X, Y):
            self.back_propagate(x, y)

        self.descend_gradient(learning_rate, batch_size = len(X))


    def stochastic_gradient_descent(self, X, Y, learning_rate = 0.3):   #added learning rate argument
        '''realize a stochastic gradient descent over a batch'''

        for x, y in zip(X, Y):
            self.back_propagate(x, y)
            self.descend_gradient(learning_rate, len(X))        


    def reset_memoization(self):
        '''reset the memoization attributes in each neuron of the network'''

        for neuron in self.neurons:
            neuron.reset_memoization()

    def reset_accumulators(self):
        '''reset the gradient accumulator attribute in each neuron of the network'''

        for neuron in self.neurons:
            neuron.reset_accumulator()

