import Neuron
import numpy as np

class Network:
    def __init__(self,neurons,inputs,outputs, expected_outputs, cost_neuron):
        self.neurons = neurons
        self.inputs = inputs
        self.outputs = outputs
        self.expected_outputs = expected_outputs
        self.cost_neuron = cost_neuron

    def propagate(self, x):
        self.reset_memoization()
        for index, input_neuron in enumerate(self.inputs):
            input_neuron.setvalue(x[index])
        y = np.zeros(len(self.outputs))
        for index, output_neuron in enumerate(self.outputs):
            y[index] = output_neuron.evaluate()
        return y

    def back_propagate(self,y):
        for index, expected_output_neuron in enumerate(self.expected_outputs):
            expected_output_neuron.set_value(y)
        cost = self.cost_neuron.evaluate()
        for input_neuron in enumerate(self.inputs):
            input_neuron.get_gradient()
        return cost

def descend_gradient(self, learning_rate, batch_size):
    for neuron in self.neurons:
        neuron.descend_gradient(learning_rate, batch_size)

    def batch_gradient_descent(self, X, Y):
        self.reset_accumulators()
        for x, y in zip(X, Y):
            self.reset_memoization()
            self.propagate(x)

    def stochastic_gradient_descent(self, X, Y):
        pass

    def reset_memoization(self):
        for neuron in self.neurons:
            neuron.reset_memoization()

    def reset_accumulators(self):
        for neuron in self.neurons:
            neuron.reset_accumulator()




