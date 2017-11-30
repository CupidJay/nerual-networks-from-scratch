import random
import numpy as np


class NN(object):
	def __init__(self, sizes):
		self.num_layers = len(sizes)-1
		self.sizes = sizes
		self.biases = [np.random.randn(size, 1) for size in sizes[1:]]
		self.weights = [np.random.randn(m, n) for m, n in zip(sizes[1:], sizes[:-1])]

	def forward(self, x):
		"""
		#this forward method used especially for testing, 
		cause when testing we don't need to store the forward activations and update parametesrs.
		"""
		for b, w in zip(self.biases, self.weights):
			z = np.dot(w, x) + b
			x = sigmoid(z)

		return x

	def backward(self, x, y):
		"""
		in fact this backward include forward once to compute activation layerwise and the final loss
		and backward once to backprop the final loss to each parameter
		"""
		delta_b = [np.zeros(b.shape) for b in self.biases]
		delta_w = [np.zeros(w.shape) for w in self.biases]

		activation = x #forward activation
		activations = [] #store the activation value for each layer
		zs = [] #store the z value for each layer

		for b, w in zip(self.biases, self.weights):
			z = np.dot(w, activation) + b
			zs.append(z)
			activation = sigmoid(z)
			activations.append(activation)

		#backward pass
		for l in range(1, self.num_layers):
			z = zs[-l]
			if l==1:
				delta = self.cost_prime(activation, y) * sigmoid_prime(z)
			else:
				delta = np.dot(self.weights[-l+1].transpose(), delta)*sigmoid_prime(z)
			delta_b[-l] = delta
			delta_w[-l] = np.dot(delta, activations[-l-1].transpose())

		return delta_b, delta_w

	def train(self, train_set, val_set, learning_rate=3, learning_rate_decay=0.95, num_epochs=15, batch_size=200):
		"""
		train the nerual network using stochastic gradient descent(SGD)

		Inputs:
		train_set: zip(X, y) where X is A numpy array of shape(N, D) giving training data
		y is A numpy array of shape(N, 10) giving training labels

		val_set: zip(X_val, y_val) where X_val is A numpy array of shape(N_val, D) giving validation data
		y_val is A numpy array of shape(N_val, ) giving validation labels

		learning_rate: Scalar giving learning rate for optimization

		epochs: Number of training epochs

		batch_size: Number of training examples to use per step

		"""
		num_train = len(train_set)
		num_val = len(val_set)
		self.learning_rate = learning_rate

		for epoch in range(num_epochs):
			print('*'*15)
			print('epoch {} / {}'.format(epoch+1, num_epochs))

			#shuffle the train_set
			random.shuffle(train_set)
			mini_batches = [train_set[k: k+batch_size] for k in range(0, num_train, batch_size)]

			for mini_batch in mini_batches:
				self.update_mini_batch(mini_batch)

			self.learning_rate *= learning_rate_decay

			print('val performance is : ', end='')
			self.test(val_set)

	def update_mini_batch(self, mini_batch):
		sum_delta_b = [np.zeros(b.shape) for b in self.biases]
		sum_delta_w = [np.zeros(w.shape) for w in self.weights]

		#this loop can be avoided so that we could use matrix multiplication
		for x, y in mini_batch:
			delta_b, delta_w = self.backward(x, y)
			sum_delta_b = [nb+b for nb, b in zip(sum_delta_b, delta_b)]
			sum_delta_w = [nw+w for nw, w in zip(sum_delta_w, delta_w)]

		alpha = self.learning_rate / len(mini_batch)
		self.biases = [b-alpha*nb for b, nb in zip(self.biases, sum_delta_b)]
		self.weights = [w-alpha*nw for w, nw in zip(self.weights, sum_delta_w)]

	def test(self, test_data):
		test_results = [(np.argmax(self.forward(x)), y) for (x, y) in test_data]
		num_test = len(test_data)
		num_correct = sum(int(x==y) for (x, y) in test_results)
		print('correct numbers {} / {}'.format(num_correct, num_test))
		

	def cost_prime(self, output, y):
		return (output - y)


def sigmoid(z):
	#sigmoid(z) = 1/(1+e^(-z))
	return 1.0 / (1.0+np.exp(-z))


def sigmoid_prime(z):
	#the deriviate of sigmoid(z)
	temp = sigmoid(z)
	return temp*(1-temp)
