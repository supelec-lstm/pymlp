from neuron import *

def test_parents_children():
	# We create a Neuron only for the purpose of testing the class
	neuron1 = Neuron('neuron1', [])
	assert neuron1.name == 'neuron1'
	neuron2 = Neuron('neuron2', [])
	neuron3 = Neuron('neuron3', [neuron1, neuron2])
	assert neuron3.parents == [neuron1, neuron2]
	assert neuron1.children == [neuron3]
	assert neuron2.children == [neuron3]

# Test the subclasses

def test_constant_gradient_neuron():
	input_neuron = InputNeuron('input')
	neuron = ConstantGradientNeuron('neuron', [input_neuron], value=1)
	# get_gradient
	assert neuron.get_gradient() == {'input': 1}
	# set_value
	neuron.set_value(5)
	neuron.reset_memoization()
	assert neuron.get_gradient() == {'input': 5}

def test_input_neuron():
	neuron = InputNeuron('neuron')
	other_neuron = InputNeuron('other_neuron', 3)
	# evaluate
	assert neuron.evaluate() == 0
	assert other_neuron.evaluate() == 3
	# set_value
	neuron.set_value(5)
	assert neuron.evaluate() == 5

def test_bias_neuron():
	neuron = BiasNeuron('neuron')
	# evaluate
	assert neuron.evaluate() == 1

def test_linear_neuron():
	input_neuron = InputNeuron('input', 2)
	neuron = LinearNeuron('neuron', [input_neuron], lambda n: np.ones(n))
	gradient_neuron = ConstantGradientNeuron('gradient', [neuron], value=1)
	# evaluate
	y = 2
	assert neuron.evaluate() == y
	#get_gradient
	assert neuron.get_gradient() == {'input': 1}

def test_sigmoid_neuron():
	input_neuron = InputNeuron('input', 2)
	neuron = SigmoidNeuron('neuron', [input_neuron], lambda n: np.ones(n))
	gradient_neuron = ConstantGradientNeuron('gradient', [neuron], value=1)
	# evaluate
	y = 1/(1+np.exp(-2))
	assert neuron.evaluate() == y
	#get_gradient
	assert neuron.get_gradient() == {'input': y*(1-y)}

def test_tanh_neuron():
	input_neuron = InputNeuron('input', 2)
	neuron = TanhNeuron('neuron', [input_neuron], lambda n: np.ones(n))
	gradient_neuron = ConstantGradientNeuron('gradient', [neuron], value=1)
	# evaluate
	y = np.tanh(2)
	assert neuron.evaluate() == y
	#get_gradient
	assert neuron.get_gradient() == {'input': 1 - y*y}

def test_relu_neuron():
	input_neuron = InputNeuron('input', 2)
	neuron = ReluNeuron('neuron', [input_neuron], lambda n: np.ones(n))
	gradient_neuron = ConstantGradientNeuron('gradient', [neuron], value=1)
	# evaluate
	assert neuron.evaluate() == 2
	#get_gradient
	assert neuron.get_gradient() == {'input': 1}

	input_neuron.set_value(-3)
	input_neuron.reset_memoization()
	neuron.reset_memoization()
	# evaluate
	assert neuron.evaluate() == 0
	#get_gradient
	assert neuron.get_gradient() == {'input': 0}

def test_softmax_neuron():
	input_neuron1 = InputNeuron('input1', 2)
	input_neuron2 = InputNeuron('input2', 3)
	neuron1 = SoftmaxNeuron('neuron1', [input_neuron1, input_neuron2], 0)
	neuron2 = SoftmaxNeuron('neuron2', [input_neuron1, input_neuron2], 1)
	gradient_neuron = ConstantGradientNeuron('gradient', [neuron1, neuron2], value=1)
	# evaluate
	total = np.exp(2) + np.exp(3)
	y1 = np.exp(2) / total
	y2 = np.exp(3) / total
	assert neuron1.evaluate() == y1
	assert neuron2.evaluate() == y2
	# get_gradient
	assert neuron1.get_gradient() == {'input1': y1*(1-y1), 'input2': -y1*y2}
	assert neuron2.get_gradient() == {'input1': -y1*y2, 'input2': y2*(1-y2)}

def test_squared_error_neuron():
	expected_output1 = InputNeuron('expected_output1', 0)
	expected_output2 = InputNeuron('expected_output2', 5)
	output_neuron1 = InputNeuron('output1', 2)
	output_neuron2 = InputNeuron('output2', 3)
	neuron = SquaredErrorNeuron('neuron', [expected_output1, expected_output2], [output_neuron1, output_neuron2])
	# evaluate
	assert neuron.evaluate() == np.sqrt(8)
	# get_gradient
	assert neuron.get_gradient() == {'output1': 4, 'output2': -4}

def test_cross_entropy_neuron():
	expected_class = InputNeuron('expected_class', 0)
	output_neuron1 = InputNeuron('output1', 0.75)
	output_neuron2 = InputNeuron('output2', 0.25)
	neuron = CrossEntropyNeuron('neuron', [expected_class], [output_neuron1, output_neuron2])
	# evaluate
	assert neuron.evaluate() == -np.log(0.75)
	# get_gradient
	assert neuron.get_gradient() == {'output1': -4/3, 'output2': 0}

	expected_class.set_value(1)
	expected_class.reset_memoization()
	neuron.reset_memoization()
	assert neuron.evaluate() == -np.log(0.25)
	# get_gradient
	assert neuron.get_gradient() == {'output1': 0, 'output2': -4}

# Small networks

def test_small_linear_network():
	input_neuron = InputNeuron('input', 2)
	neuron = LinearNeuron('neuron', [input_neuron], lambda n: np.ones(n))
	other_neuron = LinearNeuron('other_neuron', [input_neuron, neuron], lambda n: np.arange(1, n+1))
	gradient_neuron = ConstantGradientNeuron('gradient', [other_neuron], value=1)
	# evaluate
	assert neuron.evaluate() == 2
	assert other_neuron.evaluate() == 6
	# get_gradient
	assert other_neuron.get_gradient() == {'input': 1, 'neuron': 2}
	assert neuron.get_gradient() == {'input': 2}
	# check accumulators
	assert np.all(other_neuron.acc_dJdw == np.array([2, 2]))
	assert np.all(neuron.acc_dJdw == np.array([4]))

	# another inputs
	input_neuron.reset_memoization()
	neuron.reset_memoization()
	other_neuron.reset_memoization()
	gradient_neuron.reset_memoization()
	input_neuron.set_value(3)
	# evaluate
	assert neuron.evaluate() == 3
	assert other_neuron.evaluate() == 9
	# get_gradient
	assert other_neuron.get_gradient() == {'input': 1, 'neuron': 2}
	assert neuron.get_gradient() == {'input': 2}
	# check accumulators
	assert np.all(other_neuron.acc_dJdw == np.array([5, 5]))
	assert np.all(neuron.acc_dJdw == np.array([10]))

	# reset accumulators
	neuron.reset_accumulator()
	other_neuron.reset_accumulator()
	assert neuron.acc_dJdw == 0
	assert other_neuron.acc_dJdw == 0