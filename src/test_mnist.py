from neuron import *
from network import *
from mnist import *

#get_training_set() => ([[image (784)], ...], (dim1,dim2), labels)


inputs = []
for i in range(784):
    name = 'input'
    name += str(i)
    inputs.append(InputNeuron(name = name))


n = 5
m = 5
neurons = []
for i in range(n):
    name = 'hidden_0_'
    name += str(i)
    neurons.append(ReluNeuron(name = name, parents = inputs))

for i in range(1,m):
    name = 'hidden_'
    name += str(i) + '_'
    for j in range(n):
        name += str(j)
        name += str(i)
        neurons.append(ReluNeuron(name = name, parents = neurons[(784+n*(i-1)):(784+n*i)]))

output = [ReluNeuron(name = 'output', parents = neurons[-n:])]

expected_output = [InputNeuron(name = 'expected_output')]

cost_n = SquaredErrorNeuron(name = 'cost', parents = expected_output + output)

neurons = neurons + output + expected_output + [cost_n]

network = Network(neurons, inputs, output, expected_output, cost_n)


training_set = get_training_set()

Y_train = [[y] for y in training_set[2].tolist()]

network.stochastic_gradient_descent(training_set[0], Y_train, learning_rate = 0.08)

