import _pickle as cPickle
import gzip

import numpy as np

'''
the dataset of mnist of format "mnist.pkl.gz" can be download at 
http://deeplearning.net/tutorial/gettingstarted.html#index-1
'''

def load_data():
	f = gzip.open('mnist.pkl.gz')
	train_set, val_set, test_set = cPickle.load(f)
	f.close()

	return train_set, val_set, test_set

#reshape x to shape(28*28, 1) and encode y to be onehot of length 10
def load_data_wrapper():
	train_set, val_set, test_set = load_data()
	X_shape = (28*28, 1)
	train_X = [np.reshape(X, X_shape) for X in train_set[0]]
	train_y = [one_hot(y) for y in train_set[1]]

	val_X = [np.reshape(X, X_shape) for X in val_set[0]]
	val_y = val_set[1]

	test_X = [np.reshape(X, X_shape) for X in test_set[0]]
	test_y = test_set[1]

	return zip(train_X, train_y), zip(val_X, val_y), zip(test_X, test_y)

def one_hot(j):
	ans = np.zeros((10, 1))
	ans[j] = 1
	return ans
