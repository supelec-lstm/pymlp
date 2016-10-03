import numpy as np
import random
import neurone

class Network ():

    def __init__(self, neurons, expected_outputs, inputs, outputs, cost_neuron):
        self.neurons=neurons
        self.expected_outputs=expected_outputs
        self.inputs=inputs
        self.outputs=outputs
        self.cost_neuron=cost_neuron

    def propagate (self, x):

        "Insérer les entrées dans le réseau"

        for neuron in self.inputs :
            neuron.set_value(x[i])

        output_values=[]
        for output in self.outputs:
            output_values+=[self.outputs.evaluate()]

        return output_values

    def back_propagate (self, x, y):

        self.propagate(x)

        for (expec_neuron,z) in zip(self.expected_outputs, y):
            expec_neuron.set_value(z)

        self.cost_neuron.evaluate()

        for input in self.inputs:
            input.get_gradient()


    def descend_gradient (learning_rate, batch_size):
        for neuron in self.neurons:
            neuron.descend_gradient(learning_rate, batch_size)

    def batch_gradient_descent (X,Y, learning_rate):
        for (x,y) in zip(X,Y):
            self.back_propagate(x,y)

        self.descend_gradient(learning_rate, len(Y))

    def stochastic_gradient_descent (X, Y, learning_rate):
        for (x,y) in zip(X,Y):
            self.back_propagate(x, y)
            self.descend_gradient(learning_rate, len(Y))

    def reset_memoization(self):
        for neuron in self.neurons:
            neuron.reset_memorization()

    def reset_accumulators (self):
        for neuron in self.neurons:
            neuron.reset_accumulator()


