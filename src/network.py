import numpy as np
from neuron import *

class Network:
	def __init__(self, neurons, inputs, outputs, expected_outputs, cost_neuron):
		self.neurons = neurons
		self.inputs = inputs
		self.outputs = outputs
		self.expected_outputs = expected_outputs
		self.cost_neuron = cost_neuron

	def propagate(self, x):
		""" 
			Return the output of the network.
			
			:param x: The input.
			:type x: np.array
			:return: The output of the network.
			:rtype: np.array
		"""
		for i, neuron in enumerate(self.inputs):
			neuron.set_value(x[i])
		y = np.zeros(len(self.outputs))
		for i, neuron in enumerate(self.outputs):
			y[i] = neuron.evaluate()
		return y

	def back_propagate(self, y):
		"""
			Compupute the gradient of each neuron of the network.

			back_propagate must be called after propagate.

			:param y: The expected output.
			:type y: np.array
			:return: The cost computed by the cost_neuron.
			:rtype: float
		"""
		for i, neuron in enumerate(self.expected_outputs):
			neuron.set_value(y[i])
		cost = self.cost_neuron.evaluate()
		for neuron in self.inputs:
			neuron.get_gradient()
		return cost

	def descend_gradient(self, learning_rate, batch_size=1):
		"""
			Apply gradient descent on all the neurons.

			:param learning_rate: The learning rate of the gradient descent.
			:param batch_size: The number of examples used to approximate the gradient.
			:type learning_rate: float
			:type batch_size: int
		"""
		for neuron in self.neurons:
			neuron.descend_gradient(learning_rate, batch_size)

	def batch_gradient_descent(self, X, Y, learning_rate):
		"""
			Apply a batch gradient descent.

			The first dimension of X and Y must be the same.

			:param X: The inputs.
			:param Y: The expected outputs.
			:param learning_rate: The learning rate of the gradient descent.
			:type X: np.array
			:type Y: np.array
			:type learning_rate: float
		"""
		self.reset_accumulators()
		for x, y in zip(X, Y):
			self.reset_memoization()
			self.propagate(x)
			print(self.back_propagate(y))
		self.apply_gradient_descent(learning_rate, X.shape[0])

	def stochastic_gradient_descent(self, X, Y, learning_rate):
		"""
			Apply a stochastic gradient descent.

			The first dimension of X and Y must be the same.

			:param X: The inputs.
			:param Y: The expected outputs.
			:param learning_rate: The learning rate of the gradient descent.
			:type X: np.array
			:type Y: np.array
			:type learning_rate: float
		"""
		for x, y in zip(X, Y):
			self.reset_accumulators()
			self.reset_memoization()
			self.propagate(x)
			print(self.back_propagate(y))
			self.apply_gradient_descent(learning_rate)

	def reset_memoization(self):
		""" Reset memoization of all the neurons. """
		for neuron in self.neurons:
			neuron.reset_memoization()

	def reset_accumulators(self):
		""" Reset accumulators of all the neurons. """
		for neuron in self.neurons:
			neuron.reset_accumulators()