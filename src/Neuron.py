class Neuron:

    def __init__(self,name,parents,init_function):
        """Constructor used for initialising the neuron, declaring its name, its neuron parents and its weights"""
        self.name = name
        self.parents = parents
        self.w = init_function(len(parents))
        self.init_function = init_function
        self.children, self.x, self.y, self.dJdx, self.acc_dJdw = None

    def evaluate(self):
        """Return the output of the neuron as a double"""
        pass

    def get_gradient(self):
        """Return the gradient of the neuron, based on its inputs and the associated weights"""
        pass

    def descend_gradient(self,learning_rate,batch_size):
        """Updates the weights related to the neuron"""
        pass

    def add_child(self, enfant):
        """Add a child to the neuron"""
        self.children.append(enfant)

    def reset_memoization(self):
        """Reset memoization"""
        self.x = None
        self.y = None
        self.dJdx = None

    def reset_accumulator(self):
        """Reset accumulator"""
        self.acc_dJdw = None

    def activation_function(self):
        """Activation function for the neuron"""
        raise NotImplementedError(self)

    def gradient_activation_function(self):
        """Calculate the derivative of the function"""
        raise NotImplementedError(self)