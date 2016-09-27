import numpy as np
from neuron import *
from network import *

def test_linear_regression():
	input_neuron = InputNeuron('input')
	output_neuron = LinearNeuron('output', [input_neuron], lambda n: np.ones(n))
	expected_output = InputNeuron('expected_output')
	cost_neuron = SquaredErrorNeuron('cost', [expected_output], [output_neuron])
	network = Network([input_neuron, output_neuron, expected_output, cost_neuron], [input_neuron],
		[output_neuron], [expected_output], cost_neuron)
	assert network.propagate(np.array([2])) == 2
	assert network.back_propagate(np.array([5])) == 3
	network.reset_memoization()
	assert network.propagate(np.array([5])) == 5
	assert network.back_propagate(np.array([5])) == 0
