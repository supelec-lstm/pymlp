# -*- coding: utf-8 -*-
"""
Éditeur de Spyder

Ceci est un script temporaire.
"""

import numpy as np


class Neuron:
    
    def __init__(self, name, parents, init_function):
        """
        neuron initialisation
        """
        self.name = name
        self.parents = parents
        self.children = None
        for parent in parents:
            parent.add_child(self)
            self.w[parent] = init_function()
        self.x = None
        self.y = None
        self.dJdx = None
        self.accumulator_dJdw = None
    
    def evaluate(self):
        """
        evaluate the value of the utpout of the neuron
        """
        if not self.y:
            for parent in self.parents:
                self.x[parent]=self.parents[parent].evaluate()
            self.y=self.activation_function()
        return self.y
    
    def get_gradient(self):
        """
        get the value of dJdx
        """
        if not self.dJdx :
            dJdy=0
            for child in self.children:
                dJdy += child.get_gradiant()[self.name]
            h=np.dot(self.x,self.w)
            dydh=self.gradiant_activation_function(h)
            dJdx={neuron : self.w[neuron]*dJdy*dydh for neuron in self.w}
            self.accumulator_dJdw={parent : self.accumulator_dJdw[parent]+dJdy*dydh*self.x[parent] for parent in self.accumulator_dJdw}
        return dJdx
        
    def descent_gradient(self,learning_rate, batch_size):
        """
        weights updating
        """
        for parent in self.w:
            self.w[parent] -= (learning_rate / batch_size)*self.accumulator_dJdw[parent]

    
    def add_child(self,neuron):
        """
        add a child to this neuron
        """
        self.children[neuron.name]=neuron
    
    def reset_memoization(self):
        """
        reset x, y and dJdx
        """
        self.x,self.y,self.dJdx=None,None,None
    
    def reset_accumulator(self):
        """
        reset accumulator_dJdw
        """
        self.accumulator_dJdw = None


    