import argparse
import numpy as np
from datasets.data_loader import load_data_wrapper
from models.nn import NN
import os
import datetime
import pytz
import yaml
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser()
#some hyper parameters.
parser.add_argument('--lr', type=float, default=4)
parser.add_argument('--lr_decay', type=float, default=0.95)
parser.add_argument('--num_epochs', type=int, default=20)
parser.add_argument('--batch_size', type=int, default=100)

#log parameters
parser.add_argument('--save_log', type=bool, default=True)
parser.add_argument('--plot', type=bool, default=True)

here = os.path.dirname(os.path.abspath(__file__))


def get_log_dir(config, history):
	name = 'MODEL'
	cfg = dict(lr=config.lr, 
			   decay=config.lr_decay,
			   epochs = config.num_epochs,
			   batch_size=config.batch_size)

	for k, v in cfg.items():
		v = str(v)
		if '/' in v:
			continue
		name += '_%s-%s' % (k.upper(), v)

	now = datetime.datetime.now(pytz.timezone('Asia/Shanghai'))
	name += '_TIME-%s' % now.strftime('%Y%m%d-%H%M%S')

	log_dir = os.path.join(here, 'logs', name)
	if not os.path.exists(log_dir):
		os.makedirs(log_dir)
	with open(os.path.join(log_dir, 'config.yaml'), 'w') as f:
		yaml.safe_dump(cfg, f, default_flow_style=False)

	#save the loss and acc history during training to the log_dir
	loss_dir = os.path.join(log_dir, 'loss.txt')
	np.savetxt(loss_dir, history['loss_history'])
	

	train_acc_dir = os.path.join(log_dir, 'train_acc.txt')
	np.savetxt(train_acc_dir, history['train_acc_history'])

	val_acc_dir = os.path.join(log_dir, 'val_acc.txt')
	np.savetxt(val_acc_dir, history['val_acc_history'])

	if config.plot==False:
		return log_dir

	#plot the figure of log history. you can choose whether plot or not.
	loss_fig_dir = os.path.join(log_dir, 'loss.png')
	plt.figure(1)
	plt.plot(history['loss_history'], label='L2 loss')
	plt.title('loss per epoch')
	plt.xlabel('epoch')
	plt.ylabel('L2 loss')
	plt.savefig(loss_fig_dir)

	acc_fig_dir = os.path.join(log_dir, 'acc.png')
	plt.figure(2)
	plt.plot(history['train_acc_history'], color='blue', label='train acc history')
	plt.plot(history['val_acc_history'], color='red', label='val acc history')
	plt.title('train and val acc per epoch')
	plt.xlabel('epoch')
	plt.ylabel('accuracy')
	plt.savefig(acc_fig_dir)

	return log_dir

def main(config):
	train_set, val_set, test_set = load_data_wrapper()

	#define the network
	net = NN([784, 50, 10])

	#train the network
	history = net.train(train_set, val_set, config.lr, config.lr_decay, config.num_epochs, config.batch_size)

	#save log
	if config.save_log:
		log_dir = get_log_dir(config, history)
		print('log file has been save to %s'%log_dir)

	#test the network
	print('final test performance is : ', end='')
	net.test(test_set)

if __name__=="__main__":
	config = parser.parse_args()
	print(config)
	main(config)
