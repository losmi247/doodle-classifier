import numpy as np
import random

#
# Class for a network made out of layers
# of perceptrons.
#
class NeuralNetwork:
    # constructor from an array of sizes of layers
    def __init__(self, layer_sizes, data):
        self.number_of_layers = len(layer_sizes)
        self.layer_sizes = layer_sizes

        # parameters of the network
        self.biases = [np.random.randn(s) for s in layer_sizes[1:]]
        self.weights = [np.random.randn(x,y) for x,y in zip(layer_sizes[1:],layer_sizes[:-1])]
        
        # the data
        self.data = data
        
    #
    #   Methods
    #
        
    # method that feeds forwards an input a to get the network's output,
    # and returns it as the result.
    def feed_forward(self, a):
        for w,b in zip(self.weights,self.biases):
            a = np.vectorize(NeuralNetwork.sigmoid)(np.dot(w,a) + b)
        return a
    
    # method to train the neural network
    def train(self, epochs = 100, m = 100, learning_rate = 0.01):
        for i in range(epochs):
            # shuffle the training data before taking mini batches
            random.shuffle(self.data.training_set)
            mini_batches = [self.data.training_set[k*m:(k+1)*m] for k in range(len(self.data.training_set)/m)]
            for mini_batch in mini_batches:
                self.process_mini_batch(mini_batch, learning_rate)
            print("Epoch ", i, "/", epochs, " finished.")
            
    # method to process a mini batch and update parameters accordingly
    def process_mini_batch(self, mini_batch, learning_rate):
        # the gradient that we will find
        nabla_weights = [np.zeros(x,y) for x,y in zip(self.layer_sizes[1:],self.layer_sizes[:-1])]
        nabla_biases = [np.zeros(s) for s in self.layer_sizes[1:]]
        
        # go through each input in the mini batch and find gradient of Cx using backpropagation
        for x,y in mini_batch:
            nabla_weights_x, nabla_biases_x = self.backpropagate(x,y)
            
            # sum up the contributions
            nabla_weights = [nb+nbx for nb,nbx in zip(nabla_weights,nabla_weights_x)]
            nabla_biases = [nb+nbx for nb,nbx in zip(nabla_biases,nabla_biases_x)]
        
        # move by -n*gradient(C) where C is the cost function - gradient descent
        self.weights = [w-learning_rate*nw for w,nw in zip(self.weights,nabla_weights)]
        self.biases = [b-learning_rate*nb for b,nb in zip(self.biases,nabla_biases)]
        
    # method to calculate the gradient of the cost function C
    def backpropagate(self, x, y):
        pass
    
    #
    #   Static Methods
    #
    def sigmoid(x):
        return 1 / (1 + np.exp(x))
    def sigmoid_prime(x):
        y = NeuralNetwork.sigmoid(x)
        return y * (1 - y)