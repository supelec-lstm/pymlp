from network import *
from neuron import *
from random import *
import matplotlib.pyplot as plt

input1, input2 , bias1= InputNeuron(name = 'input1'), InputNeuron(name = 'input2'), InputNeuron(name = 'bias1')
hidden1, hidden2, bias2 = SigmoidNeuron(name = 'hidden1', parents = [input1, input2, bias1]), SigmoidNeuron(name = 'hidden2', parents = [input1, input2, bias1]), InputNeuron(name = 'bias2')
hidden3, hidden4, bias3 = SigmoidNeuron(name = 'hidden3', parents = [hidden1, hidden2, bias2]), SigmoidNeuron(name = 'hidden3', parents = [hidden1, hidden2, bias2]), InputNeuron(name = 'bias3')

output = SigmoidNeuron(name = 'output', parents = [hidden3, hidden4, bias3])
expected_output = InputNeuron(name = 'expected_output')
cost_n = SquaredErrorNeuron(name = 'cost', parents = [expected_output, output])


network = Network([input1, input2, bias1, bias2, bias3, hidden1, hidden2, hidden3, hidden4, output, expected_output, cost_n],
	[input1, input2, bias1, bias2, bias3], [output], [expected_output], cost_n)

batch = [[[0,0,1,1,1], [0]], [[0,1,1,1,1], [1]], [[1,0,1,1,1], [1]], [[1,1,1,1,1], [0]]]

X=[]
Y=[]
for i in range(1000000):
	n = randint(0,3)
	X.append(batch[n][0])
	Y.append(batch[n][1])

print(hidden1.w)
print(hidden2.w)
print(output.w)

costs = []

for i in range(100):

	costs += network.stochastic_gradient_descent(X[1000*i:1000*(i+1)], Y[1000*i:1000*(i+1)], 0.1)



print(hidden1.w)
print(hidden2.w)
print(output.w)

print(network.propagate(batch[0][0]))
print(network.propagate(batch[1][0]))
print(network.propagate(batch[2][0]))
print(network.propagate(batch[3][0]))

x = np.linspace(-0.5, 1.5, num = 100)
plane = np.zeros((100,100))

for i in range(100):
	for j in range(100):
		plane[i][j] = network.propagate([x[i],x[99-j],1,1])

print(plane)

plt.imshow(plane, origin = 'lower')

plt.show()

plt.plot(costs)

plt.show()