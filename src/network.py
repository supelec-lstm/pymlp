# -*- coding: utf-8 -*-
"""
Created on Tue Sep 27 15:00:20 2016

@author: Thaïs
"""

import Neuron
import numpy as np
import random

class Network:
    
    def __init__(self, neurons, inputs, outputs, expected_outputs, cost_neuron):
        self.neurons=neurons
        self.inputs=inputs
        self.outputs=outputs
        self.expected_outputs=expected_outputs
        self.cost_neuron=cost_neuron
        
    def reset_memoization(self):
        for neuron in self.neurons:
            neuron.reset_memoization()
        for neuron in self.outputs:
            neuron.reset_memoization()
        for neuron in self.inputs:
            neuron.reset_memoization()
        for neuron in self.expected_output:
            neuron.reset_memoization()
        self.cost_neuron.reset_memoization()
        
    def reset_accumulators(self):
        for neuron in self.neurons:
            neuron.reset_accumulator()
        for neuron in self.outputs:
            neuron.reset_accumulator()
        for neuron in self.inputs:
            neuron.reset_accumulator()
        for neuron in self.expected_output:
            neuron.reset_accumulator()
        self.cost_neuron.reset_accumulator()
        
    def propagate(self,data_input):
        #data_input=vecteur d'entrée
        #rentre le vecteur dans les neurones d'entrée
        if data_input.size>self.inputs.size:
            return 'Error'
        else:
            i=0
            while i<data_input.size:
                self.inputs.set_value(data_input[i])
                i=i+1
            while i<self.inputs.size:
                self.inputs.set_value(0)
                i=i+1
                
        neuron=self.neurons.copy() #calul des neurones, il faut parents déjà calculés
        while len(neuron)>0:
            for neur in neuron:
                test=True
                i=0
                while test and i<len(neuron.parents):  #verifier que children ont calculé leur gradient
                    if neuron.parents[i].y==None:
                        test=False
                    i=i+1
                if test:
                    neur.evaluate()
                    neuron.remove(neur)    
            
        output=np.zeros([len(self.outputs),1])
        for i in range(1,len(self.outputs)):
            output[i,0]=self.outputs[i].evaluate()
        self.cost_neuron.evaluate()  #ses parents sont les outputs
        return output
        
    def back_propagate(self):
        """va propager l'erreur et calculer la correction à appliquer
        sur chaque poids"""
        self.cost_neuron.calculate_gradient(self.expected_outputs,self.outputs)
        for output in self.outputs:
            output.calculate_gradient()
        neuron=self.neurons.copy()
        while len(neuron)>0:
            for neur in neuron:
                test=True
                i=0
                while test and i<len(neuron.children):  #verifier que children ont calculé leur gradient
                    if neuron.children[i].dJdx==None:
                        test=False
                    i=i+1
                if test:
                    neur.calculate_gradient()
                    neuron.remove(neur)
                    
    
    def descent_gradient(self, learning_rate, batch_size):
        """met à jour tous les poids"""
        for output in self.outputs:
            output.descent_gradient()
        for neuron in self.neurons:
            neuron.descent_gradient()
    
    def batch_gradient_descent(self, learning_rate, X, Y):
        self.reset_accumulators()
        self.reset_memoization()
        for test_input, expected_output in zip(X,Y):
            
            if expected_output.size>self.expected_outputs.size: #mise des expected outputs
                return 'Error'
            else:
                i=0
                while i<expected_output.size:
                    self.expected_outputs[i].y=expected_output[i]
                    i=i+1
                while i<self.expected_outputs.size:
                    self.expected_outputs.y=0
                    i=i+1
                    
            self.propagate(test_input)
            self.back_propagate()
            
        self.descent_gradient(learning_rate, len(X))
    
    def stochastic_gradient_descent(self, batch_size, learning_rate, X, Y):
        """choisir un batch au hasard de la taille voulue et on applique la descente du gradient"""
        Xbatch=[]
        Ybatch=[]
        Xcopy=X.copy()
        Ycopy=Y.copy()
        nb=batch_size
        while nb>0:
            i=random.randint(0,len(Xcopy)-1)
            Xbatch.append(Xcopy[i])
            Ybatch.append(Ycopy[i])
            Xcopy.pop(i)
            Ycopy.pop(i)
            nb=nb-1
        self.batch_gradient_descent(learning_rate, Xbatch, Ybatch)

            