import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(__file__, '../../src')))

import numpy as np
import matplotlib.pyplot as plt
from network import *
from neuron import *
from toy_datasets import *

def evaluate(network):
	print('(0, 0) -> ' + str(network.propagate(np.array([0, 0]))))
	print('(1, 0) -> ' + str(network.propagate(np.array([1, 0]))))
	print('(0, 1) -> ' + str(network.propagate(np.array([0, 1]))))
	print('(1, 1) -> ' + str(network.propagate(np.array([1, 1]))))

def visualize(network, n):
	x1min, x1max = -0.5, 1.5
	x2min, x2max = -0.5, 1.5
	dx1 = (x1max - x1min) / n
	dx2 = (x2max - x2min) / n
	Y = np.zeros((n, n))
	x2 = x2min
	for i in range(n):
		x1 = x1min
		for j in range(n):
			Y[i, j] = network.propagate([x1, x2])
			x1 += dx1
		x2 += dx2
	plt.imshow(Y, extent=[x1min, x1max, x2min, x2max], vmin=0, vmax=1, interpolation='none', origin='lower')
	plt.colorbar()
	plt.show()

if __name__ == '__main__':
	input_bias = BiasNeuron('b')
	input_neuron1 = InputNeuron('x1')
	input_neuron2 = InputNeuron('x2')
	inputs = [input_neuron1, input_neuron2]
	output_neuron = SigmoidNeuron('y',  [input_bias] + inputs, lambda n: np.random.rand(n) * 0.2 - 0.1)
	expected_output = InputNeuron('expected_y')
	cost_neuron = BernoulliCostNeuron('cost', [expected_output], [output_neuron])
	neurons = inputs + [input_bias, output_neuron, expected_output, cost_neuron]
	network = Network(neurons, inputs, [output_neuron], [expected_output], cost_neuron)

	X, Y = or_dataset(100)
	Y = Y.reshape((len(Y), 1))
	for _ in range(100):
		#network.stochastic_gradient_descent(X, Y, 1)
		network.batch_gradient_descent(X, Y, 1)
	print(output_neuron.w)
	evaluate(network)
	visualize(network, 100)
