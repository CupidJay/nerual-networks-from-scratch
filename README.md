# nerual-networks-from-scratch
Learn to understand and code nerual networks from scratch
I follow the work by [[Project]](http://neuralnetworksanddeeplearning.com/) 
[[Book]](https://github.com/mnielsen/neural-networks-and-deep-learning)
you can click the link above for more details.
To better understand nerual networks, I rederivate the forward and backward equations and reimplement the code.

I made modifications as below:
1. add main.py to make you easier to run the code and change hyper-parameters.
2. add Log control to save and plot your loss and acc history, which is easier for you to fine tune your network.
3. add L1 regularization and SGD with momentum and so on.

### Prerequisites
* python 3
* all the packages listed in requirements.txt(pip install requirements.txt if you need)

### Getting started
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
### Training and evaluating
1. use the parameters as default
    ```bash
    python main.py
    ```
2. set your own hyper-parameters please refer to main.py to see more information
```bash
#e.g.
python main.py --lr=0.5 --momentum=0.9 --lamda=1
```
## Finu-tune your network
* after your train your net work, the loss and acc history will be saved to the corresponding folder. see the acc and loss figure to figure out performance of your network.
#if you dont't want to save log, just set the parameter --save_log=False
1. e.g. Loss dont't decrease for a long time, maybe your learning_rate is too large.

