# nerual-networks-from-scratch
Learn to understand and code nerual networks from scratch.

I follow the work nearual-networks-and-deep-learning [[Book]](http://neuralnetworksanddeeplearning.com/) 
[[Project]](https://github.com/mnielsen/neural-networks-and-deep-learning)

you can click the link above for more information.

## Work Introduction
To better understand nerual networks, I rederivate the forward and backward equations and reimplement the code.

I made modifications as below:
* add main.py to make you easier to run the code and change hyper-parameters.
* add Log control to save and plot your loss and acc history, which is easier for you to fine tune your network.
* add L1 regularization and SGD with momentum and so on.

## Prerequisites
* python 3
* all the packages listed in requirements.txt(pip install requirements.txt if you need)

## Getting started
1. get source code
    ```bash
    git clone https://github.com/CupidJay/nerual-networks-from-scratch.git
    ```
2. download dataset
    ```bash
    cd datasets
    #download
    bash download_dataset.sh
    ```
## Training and evaluating
1. use the parameters as default
    ```bash
    python main.py
    ```
2. set your own hyper-parameters please refer to main.py to see more information
    ```bash
    #e.g.
    python main.py --lr=0.5 --momentum=0.9 --lamda=1
    ```
### Finu-tune your network
after your train your net work, the loss and acc history will be saved to the corresponding folder. see the acc and loss figure to figure out performance of your network.<br>

if you dont't want to save log, just set the parameter --save_log=False <br>

1. For example, loss dont't decrease for a long time, maybe your learning_rate is too large. <br>

2. There are many hyper-parameters for you to fine tune
* `network size`: num_layers of your network and num_nerouns of each layer
* `learning rate`: the learning rate of your optimizer(SGD e.g)
* `learning rate decay`: the rate of your learning rate decay every several epochs
* `regularization`: set it to 'l1' to choose l1 regularization and 'l2' for l2 regularization
* `lambda`: the coefficient of regularization
* `momentum`: the coefficient of momentum if your are using SGD with momentum
* `num_epochs`: the total epochs you want to train
* `batch_size`: mini-batch size

Go and have a try!
