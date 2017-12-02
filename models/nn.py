import random
import numpy as np


class NN(object):
	def __init__(self, sizes):
		self.num_layers = len(sizes)
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
		activations = [x] #store the activation value for each layer
		zs = [] #store the z value for each layer

		for b, w in zip(self.biases, self.weights):
			z = np.dot(w, activation) + b
			zs.append(z)
			activation = sigmoid(z)
			activations.append(activation)

		'''
		delta = self.cost_prime(activations[-1], y) * sigmoid_prime(zs[-1])

		delta_b[-1] = delta
		delta_w[-1] = np.dot(delta, activations[-2].transpose())
		'''
		self.loss += self.cost(activation, y)

		#backward pass
		for l in range(1, self.num_layers):
			z = zs[-l]	
			if l==1:
				delta = self.cost_prime(activation, y)*sigmoid_prime(z)
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

		learning_rate_decay: Scalar decay rate each epoch

		num_epochs: Number of training epochs

		batch_size: Number of training examples to use per step
		"""
		num_train = len(train_set)
		num_val = len(val_set)
		self.learning_rate = learning_rate

		loss_history = []
		train_acc_history = []
		val_acc_history = []

		for epoch in range(num_epochs):
			print('*'*15)
			print('epoch {} / {}'.format(epoch+1, num_epochs))

			#shuffle the train_set
			random.shuffle(train_set)
			mini_batches = [train_set[k: k+batch_size] for k in range(0, num_train, batch_size)]
			self.loss = 0
			for mini_batch in mini_batches:
				self.update_mini_batch(mini_batch)

			if epoch%5==0 and epoch!=0:
				self.learning_rate *= learning_rate_decay

			self.loss /= len(mini_batches)
			print('mean loss per mini_batch is {}'.format(self.loss))

			loss_history.append(self.loss)

			print('val performance is : ', end='')
			val_acc = self.test(val_set)
			val_acc_history.append(val_acc)

			train_acc = self.test(train_set, False)
			train_acc_history.append(train_acc)

		return {
			'loss_history': loss_history,
			'train_acc_history': train_acc_history,
			'val_acc_history': val_acc_history,
		}

	def update_mini_batch(self, mini_batch):
		sum_delta_b = [np.zeros(b.shape) for b in self.biases]
		sum_delta_w = [np.zeros(w.shape) for w in self.weights]

		#this loop can be avoided so that we could use matrix multiplication
		for x, y in mini_batch:
			delta_b, delta_w = self.backward(x, y)
			sum_delta_b = [nb+dnb for nb, dnb in zip(sum_delta_b, delta_b)]
			sum_delta_w = [nw+dnw for nw, dnw in zip(sum_delta_w, delta_w)]

		eta = self.learning_rate/len(mini_batch)
		self.biases = [b-eta*nb for b, nb in zip(self.biases, sum_delta_b)]
		self.weights = [w-eta*nw for w, nw in zip(self.weights, sum_delta_w)]	

	def test(self, test_data, verbose=True):
		if verbose:
			test_results = [(np.argmax(self.forward(x)), y) for (x, y) in test_data]
		else:
			test_results = [(np.argmax(self.forward(x)), np.argmax(y)) for (x, y) in test_data]
		num_test = len(test_data)
		num_correct = sum(int(x==y) for (x, y) in test_results)
		if verbose:
			print('correct numbers {} / {}'.format(num_correct, num_test))
		return (num_correct/num_test)*100

	def cost(self, output, y):
		return np.sum(np.square(output-y))

	def cost_prime(self, output, y):
		return (output - y)


def sigmoid(z):
	#sigmoid(z) = 1/(1+e^(-z))
	return 1.0 / (1.0+np.exp(-z))


def sigmoid_prime(z):
	#the deriviate of sigmoid(z)
	temp = sigmoid(z)
	return temp*(1-temp)
