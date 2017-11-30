import argparse
import numpy as np
from datasets.data_loader import load_data_wrapper
from models.nn import NN

parser = argparse.ArgumentParser()
#some hyper parameters.
parser.add_argument('--lr', type=float, default=3)
parser.add_argument('--lr_decay', type=float, default=0.95)
parser.add_argument('--num_epochs', type=int, default=15)
parser.add_argument('--batch_size', type=int, default=100)


def main(config):
	train_set, val_set, test_set = load_data_wrapper()

	#define the network
	net = NN([784, 50, 10])

	#train the network
	net.train(train_set, val_set, config.lr, config.lr_decay, config.num_epochs, config.batch_size)

	#test the network
	print('final test performance is : ', end='')
	net.test(test_set)

if __name__=="__main__":
	config = parser.parse_args()
	print(config)
	main(config)
