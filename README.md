# MNIST_NN


## Summary
A feedforward Neural Network was implemented in order to correctly identify and classify handwritten digits from the MNIST Database. An already prepared and preprocessed file containing the image data of the digits was provided in a csv format. This file was read and was used as a basis for training and testing the Neural Network model.

## Model Architecture
The created model has three layers, one input and output layer and a hidden layer. The input layer has 784 input nodes corresponding to the 784 pixels which are our features. The output layer has 10 artificial neurons, each corresponding to the class of the digit being recognized. The output layer has softmax as its activation function. This is a valid and optimal activation function since it spits out a probability distribution which adds up to 1. Therefore assigning different probabilities for the 10 different classes and the one with the max being what the model thinks the input likely is. The hidden layer, on the other hand, has a relu activation function, which is beneficial because it has a sparse activation and it alleviates the vanishing gradient problem.  
The number of hidden layers was decided through different trials and it was decided that having more than one hidden layer did not have any added benefits and even sometimes had a drawback.
The SparseCategoricalCrossEntropy activation function was used along with the adam optimizer function for training the model and getting accuracies. 

## Accuracies over folds

| Accuracy over five folds | 0.99 | 0.96 | 0.99 | 0.99 | 0.99 |
|--------------------------|------|------|------|------|------|
| Average Accuracy         | 0.98 |      |      |      |      |


## Learning Curves
![alt text](https://github.com/AbrehamT/MNIST_NN/blob/c8c9addeb741c8e9cf814eabc2a0dd934cd29e7f/optimal_output_graph.png)