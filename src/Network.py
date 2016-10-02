from src.InputNeuron import *
from src.SigmoidNeuron import *
from src.SquaredErrorNeuron import *
import numpy as np

class Network:

    def __init__(self,neurons, inputs, outputs, expected_outputs, cost_neuron):
        self.neurons = neurons
        self.inputs = inputs
        self.outputs = outputs
        self.expected_outputs = expected_outputs
        self.cost_neuron = cost_neuron

    def propagate(self, x):
        self.reset_memoization()
        for index, input_neuron in enumerate(self.inputs):
            input_neuron.set_value(x[index])
        y = np.zeros(len(self.outputs))
        for index, output_neuron in enumerate(self.outputs):
            y[index] = output_neuron.evaluate()
        return y

    def back_propagate(self,y):
        for index, expected_output_neuron in enumerate(self.expected_outputs):
            expected_output_neuron.set_value(y)
        cost = self.cost_neuron.evaluate()
        for input_neuron in self.inputs:
            input_neuron.get_gradient()
        return cost

    def descend_gradient(self, learning_rate, batch_size = 1):
        for neuron in self.neurons:
            neuron.descend_gradient(learning_rate, batch_size)

    def batch_gradient_descent(self,learning_rate, X, Y):
        self.reset_accumulators()
        for x, y in zip(X, Y):
            self.reset_memoization()
            self.propagate(x)
        self.descend_gradient(learning_rate, len(X))

    def stochastic_gradient_descent(self, X, Y):
        pass

    def reset_memoization(self):
        for neuron in self.neurons:
            neuron.reset_memoization()

    def reset_accumulators(self):
        for neuron in self.neurons:
            neuron.reset_accumulator()




def generateur_poids(size):
    return np.random.uniform(0,1,size)

X = [[0,0],[0,1], [1,0], [1,1]]
Y = [0,1,1,0]

input1 = InputNeuron('input1', [],generateur_poids)
input2 = InputNeuron('input2', [],generateur_poids)
hidden1 = SigmoidNeuron('hidden1', [input1, input2],generateur_poids)
hidden2 = SigmoidNeuron('hidden2', [input1, input2],generateur_poids)
output = SigmoidNeuron('output', [hidden1, hidden2],generateur_poids)
expected_outputs = InputNeuron('expected',[],generateur_poids)
cost = SquarredErrorNeuron('cost', expected_outputs,output,generateur_poids)

network = Network([hidden1, hidden2,input1,input2,output,expected_outputs,cost], [input1, input2], [output], [expected_outputs], [cost])


for j in range(0, 100):
    print("")
    for x, y in zip(X, Y):
        for i in range(len(Y)):
            network.expected_outputs[i] = Y[i]
        print('expected output :', y)
        network.propagate(x)
        print('result :',network.outputs[0].y)
    network.batch_gradient_descent( 1,X, Y)
    print('poids', hidden1.w)
    print('acc',hidden1.acc_dJdw)
    print(hidden1.x,hidden1.w)