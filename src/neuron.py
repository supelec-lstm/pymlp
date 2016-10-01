# -*- coding: utf-8 -*-
"""
Created on Mon Sep 26 23:27:58 2016

@author: Thaïs
"""

import numpy as np

class Neuron:
    
    def __init__(self, name, parents):
        self.name=name
        self.parents=parents
        self.w=np.random.randn(len(parents),1)#poids avec parents générés aléatoirement
        self.x=np.zeros([len(parents),1])  #on les met à 0 pour l'instant
        self.y=None  #sortie pas encore calculée
        self.children=[]  #liste des enfants
        self.dJdx=None  #gradient de fct cout par parents, mémoization
        self.acc_dJdw=np.zeros(self.w.shape)  #accumulation pour batch vecteur pour tous les poids
        #on va prevenir les parents
        for neuron in parents:
            neuron.add_child(self)
        
        
    def evaluate(self):
        #réévaluer les entrées
        for i in range(0,len(self.parents)):
            self.x[i,0]=self.parents[i].evaluate()
        self.y=self.activation_function(np.dot(self.w.T,self.x)[0,0])
        return self.y

        
    def calculate_gradient(self):
        """calcule le gradient et accumule"""
        net=np.dot(np.transpose(self.w),self.x)
        self.dJdx=0
        for child in self.children:
            
            #trouver le poids entre child et self
            test=True
            indice=-1 #indice de self dans parents de child
            while test:
                indice=indice+1
                if child.parents[indice]==self:
                    test=False
            self.dJdx=self.dJdx+child.get_gradient()*child.w[indice,0]*self.gradient_activation_function(net)
        self.acc_dJdw=self.acc_dJdw+self.dJdx*self.x
        
    def get_gradient(self):
        return self.dJdx
        

    def add_child(self,neuron):
        self.children.append(neuron)
        
    def reset_memoization(self):
        self.dJdx=None
        
    def reset_accumulator(self):
        self.acc_dJdw=np.zeros(self.w.shape)
        
    def activation_function(x):
        raise NotImplementedError
    
    def activation_function(x,y):
        raise NotImplementedError
    
    def gradient_activation_function(x):
        raise NotImplementedError
        
    def descent_gradient(self,learning_rate,batch_size):
        """met à jour les poids"""
        #print('zzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzz')
#        print (self.w)
#        print(self.acc_dJdw)
#        for w in self.w:
#            print('w',w)
#            print((learning_rate/batch_size)*self.acc_dJdw)
#            w=w-(learning_rate/batch_size)*self.acc_dJdw
#        print('to',self.w)
        #print(self.w)
        self.w=self.w-(learning_rate/batch_size)*self.acc_dJdw
        #print(self.w)
        
        
        
class InputNeuron(Neuron):
    """hérite de la classe Neuron"""
    def __init__(self, name, value):
        self.value=value
        self.name=name
        self.y=value  #sortie pas encore calculée
        self.children=[]  #liste des enfants
        
    def set_value(self,value):
        self.value=value
        self.y=value
        
    def activation_function(self):
        return self.value
        
    def gradient_activation_function(self,x):
        return 0
        
    def evaluate(self):
        return self.value
        
        
class LinearNeuron(Neuron):
    """garde le meme constructeur que neuron
    on définit les fct de maniere generale et on les applique dans evaluate"""
    def activation_function(x):
        return x
    def gradient_activation_function(self,x):
        return 1
        
class SigmoidNeuron(Neuron):
    def activation_function(self,x):
        return 1/(1+np.exp(-x))
    def gradient_activation_function(self,x):
        return self.activation_function(x)*(1-self.activation_function(x))
        
class TanhNeuron(Neuron):
    def activation_function(x):
        return (np.exp(x)-np.exp(-x))/(np.exp(x)+np.exp(-x))
    def gradient_activation_function(self,x):
        return 1-(TanhNeuron.activation_function(x))**2
        
class ReluNeuron(Neuron):
    def activation_function(x):
        return max(0,x)
    def gradient_activation_function(self,x):
        if x<0:
            return 0
        return 1
    
    
class SoftmaxNeuron(Neuron):
    def activation_function(vector,j):
        sum=0
        for i in range(0,len(vector)):
            sum=sum+np.exp(vector[i])
        return np.exp(vector[j])/sum
    def gradient_activation_function(vector,j):
        grad=np.zeros(vector.shape)
        for i in range (0,vector.shape[0]):
            if i!=j:
                grad[i,0]=-SoftmaxNeuron.activation_function(vector,j)*SoftmaxNeuron.activation_function(vector,i)
            else:
                grad[i,0]=SoftmaxNeuron.activation_function(vector,i)*(1-SoftmaxNeuron.activation_function(vector,i))
        return grad
    
class LeastSquareNeuron(Neuron):
    """va servir pour le cost_neuron"""
    def __init__(self, name, parents):
        self.name=name
        self.parents=parents
        self.w=np.ones([len(parents),1])#poids à 1, il faut pour la backprop
        self.x=np.zeros([len(parents),1])  #on les met à 0 pour l'instant
        self.y=None  #sortie pas encore calculée
        self.children=None #liste des enfants
        self.dJdx=None  #gradient de fct cout par parents, mémoization
        self.acc_dJdw=np.zeros(self.w.shape)  #accumulation pour batch vecteur pour tous les poids
        #on va prevenir les parents
        for neuron in parents:
            neuron.add_child(self)
    def activation_function(self,expected_output,outputs):  #listes en paramètres
        outputvect=np.zeros([len(outputs),1])
        for i in range(0,len(outputs)):
            outputvect[i,0]=outputs[i].y
        return 0.5*(np.dot((expected_output-outputvect).T,expected_output-outputvect)[0,0])
    def gradient_activation_function(self, expected_output,output):
        outputvect=np.zeros([len(output),1])
        for i in range(0,len(output)):
            outputvect[i,0]=output[i].y
        return outputvect-expected_output
    def calculate_gradient(self,expected_output,output):
        """calcule le gradient et accumule"""
#        if len(self.children)==0:
#            return
#        child=self.children[0]
#        #trouver le poids entre child et self
#        test=True
#        indice=0 #indice de self dans parents de child
#        while test:
#            if child.parents[indice]==self:
#                test=False
#            indice=indice+1
#        self.dJdx=child.get_gradient()*child.w[indice,0]*self.gradient_activation_function(expected_output,output)
#        self.acc_dJdw=self.acc_dJdw+self.dJdx*self.x
        self.dJdx=self.gradient_activation_function(expected_output, output)
        
    def evaluate(self, expected_outputs, outputs):  #outputs liste
        self.y=self.activation_function(expected_outputs, outputs)
        return self.y
        
class CrossEntropyNeuron(Neuron):
    """pas fonctionnel"""
    def crossEntropy(expected_output,output):
        return expected_output*np.log(output)+(1-expected_output)*np.log(1-output)
    def gradient_activation_function(expected_output,output):
        return None
        
    