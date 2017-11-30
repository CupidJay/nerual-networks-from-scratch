import random
import numpy as np


class NN(object):
	def __init__(self, sizes):
		self.layers = len(sizes)
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

	def train(self, train_set, val_set, learning_rate=1e-3, epochs=2, batch_size=200):
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

		for epoch in range(epochs):
			print('*'*15)
			print('epoch {} / {}'.format(epoch+1, epochs))

			#shuffle the train_set
			random.shuffle(train_set)
			mini_batches = [train_set[k: k+batch_size] for k in range(0, num_train, batch_size)]

			for mini_batch in mini_batches:
				self.update_mini_batch(mini_batch)

			print('val performance is correct {} / {}'.format(self.test(val_set), num_val))

	def update_mini_batch(self, mini_batch):
		return

	def test(self, test_data):
		test_results = [(np.argmax(self.forward(x)), y) for (x, y) in test_data]
		return sum(int(x==y) for (x, y) in test_results)



def sigmoid(z):
	#sigmoid(z) = 1/(1+e^(-z))
	return 1.0 / (1.0+np.exp(-z))


def sigmoid_prime(z):
	#the deriviate of sigmoid(z)
	temp = sigmoid(z)
	return temp*(1-temp)
