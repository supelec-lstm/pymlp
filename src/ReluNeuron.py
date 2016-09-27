class ReluNeuron(Neuron):

    def activation_function(self,x):
        """Activation function for the neuron"""
        if (x>0):
            return x
        else :
            return 0

    def gradient_activation_function(self,x):
        """Calculate the derivative of the function"""
        if (x==0):
            return None
        if (x > 0):
            return 1
        if (x < 0):
            return 0


