import argparse
import numpy as np
from datasets.data_loader import load_data_wrapper

parser = argparse.ArgumentParser()

def test_load():
	train_set, val_set, test_set = load_data_wrapper()
	for X, y in train_set:
		print('total train_X length is {}'.format(len(train_set)))
		print('train_X size is {} and train_y size is {}'.format(np.array(X).shape, np.array(y).shape))
		break

	for X, y in val_set:
		print('total val_X length is {}'.format(len(val_set)))
		print('val_X size is {} and val_y size is {}'.format(np.array(X).shape, np.array(y).shape))
		break

	for X, y in test_set:
		print('total train_X length is {}'.format(len(test_set)))
		print('train_X size is {} and train_y size is {}'.format(np.array(X).shape, np.array(y).shape))
		break


def main(config):
	test_load()
	


if __name__=="__main__":
	config = parser.parse_args()
	print(config)
	main(config)
