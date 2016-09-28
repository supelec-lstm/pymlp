from network import *
from neuron import *
from random import *


input1, input2 , bias1= InputNeuron(name = 'input1'), InputNeuron(name = 'input2'), InputNeuron(name = 'bias1')
hidden1, hidden2, bias2 = ReluNeuron(name = 'hidden1', parents = [input1, input2, bias1]), ReluNeuron(name = 'hidden2', parents = [input1, input2, bias1]), InputNeuron(name = 'bias2')
output = SigmoidNeuron(name = 'output', parents = [hidden1, hidden2, bias2])
expected_output = InputNeuron(name = 'expected_output')
cost_n = SquaredErrorNeuron(name = 'cost', parents = [expected_output, output])


network = Network([input1, input2, bias1, bias2, hidden1, hidden2, output, expected_output, cost_n],
	[input1, input2, bias1, bias2], [output], [expected_output], cost_n)

batch = [[[0,0,1,1], [0]], [[0,1,1,1], [1]], [[1,0,1,1], [1]], [[1,1,1,1], [0]]]

X=[]
Y=[]
for i in range(100000):
	n = randint(0,3)
	X.append(batch[n][0])
	Y.append(batch[n][1])

print(hidden1.w)
print(output.w)

network.stochastic_gradient_descent(X, Y, 0.8)

print(hidden1.w)
print(output.w)

print(network.propagate(batch[0][0]))
print(network.propagate(batch[1][0]))
print(network.propagate(batch[2][0]))
print(network.propagate(batch[3][0]))