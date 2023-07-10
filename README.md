# Multilayer Perceptron
This project implements a [multilayer perceptron](https://en.wikipedia.org/wiki/Multilayer_perceptron) (feedforward artificial neural network) from scratch in order to predict whether a cancer is malignant or benign on a dataset of
breast cancer diagnosis in the Wisconsin.  
It is divided in two parts:
- The [Training](#Training) part will train the neural network
- The [Prediction](#Prediction) part will make predictions using the trained neural network

## Training
The neural network is composed of 4 layers:
- The input layer containing 30 neurons
- A first hidden layer containing 21 neurons
- A second hidden layer containg 21 neurons
- The output layer containing 2 neurons

The neural network is trained using [backpropagation](https://en.wikipedia.org/wiki/Backpropagation) and [gradient descent](https://en.wikipedia.org/wiki/Gradient_descent), and can [early stop](https://en.wikipedia.org/wiki/Early_stopping) if necessary in order to avoid [overfitting](https://en.wikipedia.org/wiki/Overfitting).  

The program takes a dataset in input, splits it into a training part and a validation part, and then trains the neural network while displaying the loss and validation loss at each step as well as a graph showing their evolution during the learning process at the end.  
The model is saved (network topology and weights) at the end of the execution.

### How to run
From the root of the repository run `python3 mp_train.py resources/data.csv`.

### Example
![multilayer_perceptron_training](https://github.com/Git-Math/multilayer_perceptron/assets/11985913/952ccd65-5a7f-45a2-b584-00d1676257c3)

## Prediction
The program will load the weights learned in the [Training](#Training) part, perform a ion on a given set, then evaluate it
using the binary cross-entropy error function as well as some other metrics.

### How to run
From the root of the repository run `python3 mp_.py resources/data.csv`.

### Example
![multilayer_perceptron_predicting](https://github.com/Git-Math/multilayer_perceptron/assets/11985913/96703ad2-13fc-4c5d-8c76-67c65e965a71)
