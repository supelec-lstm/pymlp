import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(__file__, '../../src')))

import numpy as np
import matplotlib.pyplot as plt
from network import *
from neuron import *
from toy_datasets import *

def init_function(n):
	return np.random.rand(n) * 1 - 0.5

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

def simple_network():
	input_bias = BiasNeuron('b1')
	input_neuron1 = InputNeuron('x1')
	input_neuron2 = InputNeuron('x2')
	input_layer = [input_bias, input_neuron1, input_neuron2]
	inputs = [input_neuron1, input_neuron2]

	hidden_bias = BiasNeuron('b2')
	hidden_neuron1 = ReluNeuron('relu1', input_layer, init_function)
	hidden_neuron2 = ReluNeuron('relu2', input_layer, init_function)
	hidden_layer = [hidden_bias, hidden_neuron1, hidden_neuron2]

	output_neuron = SigmoidNeuron('y',  hidden_layer, init_function) 
	expected_output = InputNeuron('expected_y')
	cost_neuron = BernoulliCostNeuron('cost', [expected_output], [output_neuron])

	neurons = input_layer + hidden_layer + [output_neuron, expected_output, cost_neuron]
	return Network(neurons, inputs, [output_neuron], [expected_output], cost_neuron)

def fully_connected_network(layers):
	input_bias = BiasNeuron('b1')
	input_neuron1 = InputNeuron('x1')
	input_neuron2 = InputNeuron('x2')
	input_layer = [input_bias, input_neuron1, input_neuron2]
	inputs = [input_neuron1, input_neuron2]

	neurons = [] + input_layer
	parent_layer = input_layer
	for i, size in enumerate(layers):
		cur_layer = [BiasNeuron('b{}'.format(i))]
		for j in range(size):
			cur_layer.append(ReluNeuron('relu-{}-{}'.format(i, j), parent_layer, init_function))
		parent_layer = cur_layer
		neurons += cur_layer

	output_neuron = SigmoidNeuron('y',  parent_layer, init_function) 
	expected_output = InputNeuron('expected_y')
	cost_neuron = BernoulliCostNeuron('cost', [expected_output], [output_neuron])

	neurons += [output_neuron, expected_output, cost_neuron]
	return Network(neurons, inputs, [output_neuron], [expected_output], cost_neuron)

if __name__ == '__main__':
	network = fully_connected_network([4, 4])

	X, Y = xor_dataset(100)
	Y = Y.reshape((len(Y), 1))
	for _ in range(1000):
		#network.stochastic_gradient_descent(X, Y, 0.000001)
		network.batch_gradient_descent(X, Y, 0.3)
	evaluate(network)
	visualize(network, 100)