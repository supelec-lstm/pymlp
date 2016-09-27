class LinearNeuron(Neuron):

    def activation_function(self,x):
        """Activation function for the neuron"""
        return x

    def gradient_activation_function(self,x):
        """Calculate the derivative of the function"""
        return 1


