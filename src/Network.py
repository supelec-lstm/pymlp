class Network:
    def __init__(self,neurons,inputs,outputs, expected_outputs, cost_neuron):
        self.neurons = neurons
        self.inputs = inputs
        self.outputs = outputs
        self.expected_outputs = expected_outputs
        self.cost_neuron = cost_neuron

    def propagate(self, x):
        pass

    def back_propagate(self,y):
        pass

    def descend_gradient(self, learning_rate, batch_size):
        pass

    def batch_gradient_descent(self, X, Y):
        pass

    def stochastic_gradient_descent(self, X, Y):
        pass

    def reset_memoization(self):
        pass

    def reset_accumulators(self):
        pass




