import numpy as np
from src.Neuron import *

class SquarredErrorNeuron(Neuron):
    def __init__(self, name, expected_outputs, outputs,init_function):
        Neuron.__init__(self, name, [expected_outputs] + [outputs],init_function)
        self.expected_outputs = expected_outputs
        self.outputs = outputs

    def activation_function(self):
        self.expected_y = self.x[:len(self.expected_outputs)]
        self.predicted_y = self.x[len(self.outputs):]
        return np.linalg.norm(self.expected_y - self.predicted_y)

    def get_gradient(self):
        if self.dJdx is None:
            dJdy = 0
            for child in self.children:
                gradient = child.get_gradient()
                dJdy += gradient[self.name]
            self.dJdx = {}
            for index, parent in enumerate(self.outputs):
                self.dJdx[parent.name] = 2 * (self.predicted_y[index] - self.expected_y[index])
        return self.dJdx







