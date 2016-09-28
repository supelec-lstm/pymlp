# -*- coding: utf-8 -*-
"""
Created on Wed Sep 28 17:19:30 2016

@author: laurent
"""

import Neuron

class Network:
    
    def __init__(self, neurons, inputs, outputs, expected_outputs, cost_neuron):
        self.neurons = neurons
        self.inputs = inputs
        self.outputs = outputs
        self.expected_outputs = expected_outputs
        self.cost_neuron = cost_neuron
    
    def propagate(self, x):
        """
        return the values of the outputs
        """
        for neuron in self.inputs:
            neuron.set_value(x[neuron])
        value={}
        for neuron in self.outputs:
            value[neuron.name]=neuron.evaluate()
        return value
    
    def back_propagate(self, y):
        """
        calculate the gradiant of every neuron of the network
        """
        for neuron in self.outputs:
            neuron.y=y[neuron]
        gradient={}
        for neuron in self.inputs:
            gradient[neuron.name]=neuron.get_gradient()
        return gradient
        
    def descend_gradient(self, learning_rate, batch_size):
        """
        apply gradiant descent on the network
        """
        for neuron in self.neurons:
            neuron.descend_gradient()
    
    def batch_gradient_descent(self, X, Y, learnig_rate):
        """
        apply batch gradient descent
        """
        self.reset_accumulators() 
        for x in X:
            self.reset_memoization() 
            self.propagate(x) 
            for y in Y:
                self.back_propagate(y) 
        self.descend_gradient(learning_rate, len(X)) 
                
        
    def stochastic_gradient_descent(self, X, Y, learnig_rate):
        """
        apply stochastic gradient descent
        """
        for x in X:
            self.propagate(x) 
            for y in Y:
                self.reset_memoization() 
                self.reset_accumulators() 
                self.back_propagate(y) 
                self.descend_gradient(learning_rate, 1) 
        
        
    def reset_memoization(self):
        """
        reset every memoization
        """
        for neuron in self.neurons:
            neuron.reset_memoization()
        
        
    def reset_accumulator(self):
        """
        reset every acumulators
        """
        for neuron in self.neurons:
            neuron.reset_accumulators()