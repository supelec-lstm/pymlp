# -*- coding: utf-8 -*-
"""
Created on Tue Sep 27 15:00:20 2016

@author: Thaïs
"""

#import Neuron
from neuron import *

import numpy as np
import random
import matplotlib.pyplot as plt

class Network:
    
    def __init__(self, neurons, inputs, outputs, expected_outputs, cost_neuron):
        """A network is defined by a list of input neurons, a list of outputs, a list of
        hidden neurons and a cost neuron at the end"""   
        self.neurons=neurons
        self.inputs=inputs
        self.outputs=outputs
        self.expected_outputs=expected_outputs
        self.cost_neuron=cost_neuron
        
    def reset_memoization(self):
        """reset the variable used for the memoization in each neuron"""
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
        #data_input=the input vector
        #put the input in the inputs neurons
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
                
        neuron=self.neurons.copy() #calulate the output of each neuron, but the parents have to be evaluated
        while len(neuron)>0:
            for neur in neuron:
                test=True
                i=0
                while test and i<len(neur.parents):  #check if the parents are already evaluated
                    if neur.parents[i].y==None:
                        test=False
                    i=i+1
                if test:
                    neur.evaluate()
                    neuron.remove(neur)    
        for i in range(0,len(self.outputs)):
            self.outputs[i].evaluate()
        self.cost_neuron.evaluate(self.expected_outputs,self.outputs)#its parents are the outputs neurons ent the expected outputs

        
    def back_propagate(self):
        """propagate the error and calculate the correction to apply on each weight"""
        self.cost_neuron.calculate_gradient(self.expected_outputs,self.outputs)
        for output in self.outputs:
            output.calculate_gradient()
        neuron=self.neurons.copy()
        while len(neuron)>0:
            for neur in neuron:
                test=True
                i=0
                while test and i<len(neur.children):  #check if the children have already calculated their gradient
                    if neur.children[i].dJdx==None:
                        test=False
                    i=i+1
                if test:
                    neur.calculate_gradient()
                    neuron.remove(neur)
                    
    
    def descent_gradient(self, learning_rate, batch_size):
        """update all the weights"""
        for output in self.outputs:
            output.descent_gradient(learning_rate, batch_size)
        for neuron in self.neurons:
            neuron.descent_gradient(learning_rate, batch_size)
        #pas pour le cost
            
    def batch_gradient_descent(self, learning_rate, X, Y):
        self.reset_accumulators()
        self.reset_memoization()
        for test_input, expected_output in zip(X,Y):
            
            if expected_output.size>self.expected_outputs.size: #update the vector of expected outputs
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
        """choose randomly a batch with the wanted size, and apply the gradient descent to this batch"""
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
Test with XOR
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

time=[]
error=[]

#test on only one training input
#x=X[1]
#y=Y[1]
#network.expected_outputs[0,0]=y
#print('expected',y)
#network.propagate(x)
#print('output',o.y)
#print('ow',o.w)
#print('cost',cost.y)
#network.batch_gradient_descent(0.5,[x],[y])
#print('odJdx',o.dJdx)
#print('hy',h1.y,h2.y,h3.y)
#print('odJdw',o.acc_dJdw)
#print('ow2',o.w)
#network.propagate(x)
#print(o.y)
#print('cost2',cost.y)



for compt in range(0,10000):
    err=0
    if compt%1000==0:
        print("")
        print(compt)
    for x,y in zip(X,Y):
        for i in range(0,len(y)):
            network.expected_outputs[i,0]=y[i,0]
        network.propagate(x)
        err=err+cost.evaluate(network.expected_outputs,network.outputs)
        if compt%1000==0:
            print('expected',y)
            print(network.outputs[0].y)
    time.append(compt)
    error.append(err/len(X))
    network.batch_gradient_descent(0.8,X,Y)
    
    
plt.plot(time,error)
plt.title('cost function (10 000 epochs, eta=0.8)')
plt.xlabel('epochs')
plt.ylabel('average costfunction')
plt.show()
   
   
"""
start :
expected [[0]]
0.781081502232
expected [[1]]
0.818883446522
expected [[1]]
0.790651229024
expected [[0]]
0.810529615909

end :
expected [[0]]
0.0320746907204
expected [[1]]
0.980235799406
expected [[1]]
0.980208927052
expected [[0]]
0.00762835507233
"""


            