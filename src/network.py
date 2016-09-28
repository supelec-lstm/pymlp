# -*- coding: utf-8 -*-
"""
Created on Tue Sep 27 15:00:20 2016

@author: Thaïs
"""

#import Neuron
from neuron import *

#import InputNeuron
#import SigmoidNeuron
#import LeastSquareNeuron
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
        self.cost_neuron.reset_memoization()
        
    def reset_accumulators(self):
        for neuron in self.neurons:
            neuron.reset_accumulator()
        for neuron in self.outputs:
            neuron.reset_accumulator()
        self.cost_neuron.reset_accumulator()
        
    def propagate(self,data_input):
        #data_input=vecteur d'entrée
        #rentre le vecteur dans les neurones d'entrée
        if data_input.size>len(self.inputs):
            return 'Error'
        else:
            i=0
            while i<data_input.size:
                self.inputs[i].set_value(data_input[i])
                i=i+1
            while i<len(self.inputs):
                self.inputs[i].set_value(0)
                i=i+1      
        neuron=self.neurons.copy() #calul des neurones, il faut parents déjà calculés
        while len(neuron)>0:
            for neur in neuron:
                test=True
                i=0
                while test and i<len(neur.parents):  #verifier que parents ont calculé leur gradient
                    if neur.parents[i].y==None:
                        test=False
                    i=i+1
                if test:
                    neur.evaluate()
                    neuron.remove(neur)    
        for i in range(0,len(self.outputs)):
            self.outputs[i].evaluate()
        self.cost_neuron.evaluate(self.expected_outputs,self.outputs)#ses parents sont les outputs

        
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
                while test and i<len(neur.children):  #verifier que children ont calculé leur gradient
                    if neur.children[i].dJdx==None:
                        test=False
                    i=i+1
                if test:
                    neur.calculate_gradient()
                    neuron.remove(neur)
                    
    
    def descent_gradient(self, learning_rate, batch_size):
        """met à jour tous les poids"""
        for output in self.outputs:
            output.descent_gradient(learning_rate, batch_size)
        for neuron in self.neurons:
            neuron.descent_gradient(learning_rate, batch_size)
    
    def batch_gradient_descent(self, learning_rate, X, Y):
        self.reset_accumulators()
        self.reset_memoization()
        for test_input, expected_output in zip(X,Y):
            
            if expected_output.size>self.expected_outputs.size: #mise des expected outputs
                return 'Error'
            else:
                i=0
                while i<expected_output.size:
                    self.expected_outputs[i,0]=expected_output[i,0]
                    i=i+1
                while i<self.expected_outputs.size:
                    self.expected_outputs[i,0].y=0
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
        

"""
Test avec XOR
"""
i1=InputNeuron('i1',0)
i2=InputNeuron('i2',0)
h1=SigmoidNeuron('h1',[i1,i2])
h2=SigmoidNeuron('h2',[i1,i2])
h3=SigmoidNeuron('h3',[i1,i2])
o=SigmoidNeuron('o',[h1,h2,h3])
cost=LeastSquareNeuron('cost',[o])
expected_output=np.zeros([1,1])

network=Network([h1,h2,h3],[i1,i2],[o],expected_output,cost)
X=[np.array([[0],[0]]),np.array([[0],[1]]),np.array([[1],[0]]),np.array([[1],[1]])]
Y=[np.array([[0]]),np.array([[1]]),np.array([[1]]),np.array([[0]])]

for compt in range(0,100000):
    if compt%10000==0:
        print("")
    for x,y in zip(X,Y):
        for i in range(0,len(y)):
            network.expected_outputs[i,0]=y[i,0]
        network.propagate(x)
        if compt%10000==0:
            print('attendu',y)
            print(network.outputs[0].y)
    network.batch_gradient_descent(1,X,Y)
    #print('w',h1.w)
    
    
"""
début :
attendu [[0]]
0.781081502232
attendu [[1]]
0.818883446522
attendu [[1]]
0.790651229024
attendu [[0]]
0.810529615909

fin :
attendu [[0]]
0.0320746907204
attendu [[1]]
0.980235799406
attendu [[1]]
0.980208927052
attendu [[0]]
0.00762835507233
"""


            