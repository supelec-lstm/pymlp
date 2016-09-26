import numpy as np

class Neuron:
    """
        Neuron is an abstract to represent neuron.
        You should use predefined neurons of create a new class.
        
        Moreover, Neuron is a low-level class, you should only initialize neuron.
        Then use a Network to manipulate them.

        Please don't modify manually the attributes.
    """

    def __init__(self, name, parents=None, init_function=None):
        """
            Initialize a neuron.

            :param name: Name of the neuron, it is useful to retrieve its contribution to the gradient of its children.
            :param parents: Neurons from which the neuron takes its inputs.
            :param init_function: A function which takes an integer n as parameter and return a np.array of size n.
            :type name: str
            :type parents: list
            :type init_function: function
        """
        # Parameters of the neuron
        self.name = name
        self.activation_function = activation_function
        self.gradient_activation_function = gradient_activation_function
        self.children = []
        self.parents = parents or []
        self.w = init_function(len(parents)) if init_function else np.random.rand(len(parents))

        # Update parents
        for parent in parents:
            parent.add_child(self)

        # Attributes for memoization
        self.x = None
        self.y = None
        self.dJdx = None

        # Gradient's accumulator
        self.acc_dJdw = 0

    def evaluate(self):
        """
            Return the output of the neuron.

            To compute the output, we use the formula:
                
            .. math::

                    y = activation\\_function(w^Tx)


            Be careful, the neuron memorize the output in order
            to avoid computing several times the same output.
            Reset the memoization to compute another output.

            :return: The output of the neuron.
            :rtype: double
        """
        if not self.y:
            # Retrieve the inputs
            self.x = np.zeros(len(self.parents))
            for i, parent in enumerate(self.parents):
                self.x[i] = parent.evaluate()
            # Compute the output
            self.y = self.activation_function()
        return self.y

    def get_gradient(self):
        """
            Return dJ/dx and add dJ/dw to the accumulator.

            Assuming the neuron computes the output y by the formulas:

            .. math::
                
                h = w^Tx

                y = activation\\_function(h)

            The function computes the derivatives by using these formulas:

            .. math::

                \\frac{\\partial J}{\\partial h} = \\frac{\\partial J}{\\partial y} \\frac{\\partial y}{\\partial h}

                \\frac{\\partial J}{\\partial x} = \\frac{\\partial J}{\\partial h} \\frac{\\partial h}{\\partial x}

                \\frac{\\partial J}{\\partial w} = \\frac{\\partial J}{\\partial h} \\frac{\\partial h}{\\partial w}

            Be careful, the function uses the memorized x and y to compute the derivatives.

            :return: The gradient with respect to its inputs.
            :rtype: dict
        """
        if not self._dJdx:
            # Compute dJ/dy
            dJdy = 0
            for child in self.children:
                gradient = child.get_gradient()
                dJdy += gradient[self.name]
            # Compute dJ/dh
            dJdh = dJdy * self.gradient_activation_function()
            # Compute dJ/dx
            self.dJdx = {}
            for i, parent in enumerate(self.parents):
                self.dJdx[parent.name] = dJdh * self.w[i]
            self.acc_dJdw += dJdh * self.x
        return self.dJdx

    def descend_gradient(self, learning_rate, batch_size=1):
        """ 
            Adjust the weights.

            The function updates the weights by using these formulas:

            .. math::

                    \\frac{\\Delta J}{\\Delta w} = \\frac{acc\\_dJdw}{batch\\_size}

                    w(t+1) = w(t) - learning\\_rate  \\frac{\\Delta J}{\\Delta w}

        """
        self.w -= learning_rate / batch_size * self.acc_dJdw

    def reset_memoization(self):
        """ Reset the values of x, y and dJdx which have been memorized. """
        self.x = None
        self.y = None
        self.dJdx = None

    def reset_accumulator(self):
        """ Reset to 0 the accumulator of gradient with respect to w. """
        self.acc_dJdw = 0

    def add_child(self, neuron):
        """
            Add a child to the list of children.

            :param neuron: The neuron to add to the children.
            :type neuron: Neuron
        """
        self.children.append(neuron)

    def activation_function():
        raise NotImplementedError()

    def gradient_activation_function():
        raise NotImplementedError()
        
"""class ConstantNeuron(Neuron):
    def __init__(self, name, parents=None, init_function=None, value=0):
        Neuron.__init__(self, parents, init_function, lambda x: value, lambda x, y: 0)

    def set_value(value):
        self.activation_function = lambda x: value

class SigmoidNeuron(Neuron):
    def __init__(self, name, parents=None, init_function=None):
        Neuron.__init__(self, parents, init_function, lambda x: value, lambda x, y: 0)

class ReluNeuron(Neuron):
    def __init__(self, name, parents=None, init_function=None):
        Neuron.__init__(self, parents, init_function, lambda x: max(0, x), lambda x, y: 1 if x >= 0 else 0)

class SoftmaxNeuron(Neuron):
    def __init__(self, name, parents=None, init_function=None, i_input):
        Neuron.__init__(self, parents, init_function)"""