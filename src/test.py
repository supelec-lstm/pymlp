from network import *
from neuron import *
from random import *


input1, input2 = InputNeuron(name = 'input1'), InputNeuron(name = 'input2')
hidden1, hidden2 = SigmoidNeuron(name = 'hidden1', parents = [input1, input2]), SigmoidNeuron(name = 'hidden2', parents = [input1, input2])
output = SigmoidNeuron(name = 'output', parents = [hidden1, hidden2])
expected_output = InputNeuron(name = 'expected_output')
cost_n = SquaredErrorNeuron(name = 'cost', parents = [expected_output, output])

print(len(hidden1.parents))


network = Network([input1, input2, hidden1, hidden2, output, expected_output, cost_n],
	[input1, input2], [output], [expected_output], cost_n)

batch = [[[0,0], [0]], [[0,1], [1]], [[1,0], [1]], [[1,1], [0]]]

X=[]
Y=[]
for i in range(500):
	n = randint(0,3)
	X.append(batch[n][0])
	Y.append(batch[n][1])


network.stochastic_gradient_descent(X, Y, 0.05)