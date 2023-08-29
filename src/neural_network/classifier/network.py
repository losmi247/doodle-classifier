import numpy as np
import random
from src.neural_network.classifier.cost_functions import *

#
#   Utility Methods
#

# can be applied to a numpy array
def sigmoid(x):
    # clip the values to avoid overflow and vanishing gradients
    x = np.clip(x, -5, 5)
    return 1.0 / (1.0 + np.exp(-x))

# can be applied to a numpy array
def sigmoid_prime(x):
    y = sigmoid(x)
    return y * (1 - y)  # * is element-wise product

#
# Class for a network made out of layers
# of perceptrons.
#
class NeuralNetwork:
    # constructor from an array of sizes of layers
    def __init__(self, layer_sizes, data):
        self.number_of_layers = len(layer_sizes)
        self.layer_sizes = layer_sizes

        # initialise parameters of the network - weights and biases
        self.initialise_variance_corrected_parameters()

        # the data
        self.data = data

        # choose the cost function used
        self.cost_function = MeanSquaredError
        
    #
    #   Methods
    #
        
    # method that feeds forwards an input a to get the network's output,
    # and returns it as the result.
    def feed_forward(self, a):
        for w,b in zip(self.weights,self.biases):
            a = sigmoid(np.dot(w,a) + b)
        return a
    
    #
    #   Training Methods
    #
    
    # method to train the neural network
    def train(self, epochs = 30, m = 20, learning_rate = 0.001):
        accuracies = []
        cost_functions = []

        for i in range(epochs):
            # shuffle the training data before taking mini batches
            random.shuffle(self.data.training_set)
            # make the mini batches
            mini_batches = [self.data.training_set[k*m:(k+1)*m] for k in range(len(self.data.training_set)//m)]

            correct_decisions = 0
            for mini_batch in mini_batches:
                correct_decisions += self.process_mini_batch(mini_batch, learning_rate)

            print("Epoch ", i+1, "/", epochs, " finished.", end=" ")
            print("Accuracy on training set: ", correct_decisions, "/", len(self.data.training_set), end=", ")
            accuracies.append(correct_decisions/len(self.data.training_set))
            cost = self.cost_function.calc_cost_function(self)
            print("Cost function: ", cost)
            cost_functions.append(cost)

        # report the accuracy and cost function after each epoch
        return accuracies, cost_functions
            
    # method to process a mini batch and update parameters accordingly
    def process_mini_batch(self, mini_batch, learning_rate):
        # the gradient that we will find
        nabla_weights = [np.zeros((x,y)) for x,y in zip(self.layer_sizes[1:],self.layer_sizes[:-1])]
        nabla_biases = [np.zeros(s) for s in self.layer_sizes[1:]]

        # measure accuracy
        correct_decisions = 0
        
        # go through each input in the mini batch and find gradient of Cx using backpropagation
        for x,y in mini_batch:
            nabla_weights_x, nabla_biases_x, correct = self.backpropagate(x,y)

            correct_decisions += correct
    
            # sum up the contributions
            nabla_weights = [nb+nbx for nb,nbx in zip(nabla_weights,nabla_weights_x)]
            nabla_biases = [nb+nbx for nb,nbx in zip(nabla_biases,nabla_biases_x)]
        
        # move by -n*gradient(C) where C is the cost function - gradient descent
        self.weights = [w-(learning_rate/len(mini_batch))*nw for w,nw in zip(self.weights,nabla_weights)]
        self.biases = [b-(learning_rate/len(mini_batch))*nb for b,nb in zip(self.biases,nabla_biases)]

        # report number of correct decisions in mini batch
        return correct_decisions
        
    # method to calculate the gradient of the cost function C
    def backpropagate(self, x, y):
        # weighted inputs
        z = [np.zeros(s) for s in self.layer_sizes[1:]]
        # activations
        a = [np.zeros(s) for s in self.layer_sizes[1:]] 
        # first do a forward pass to calculate weighted inputs and activations
        l = 0
        for w,b in zip(self.weights,self.biases):
            if l == 0:
                z[l] = np.dot(w,x)+b
            else:
                z[l] = np.dot(w,z[l-1])+b
            a[l] = sigmoid(z[l])
            l += 1

        correct = 1 if np.argmax(a[-1]) == y else 0

        # the derivatives by z
        delta = [np.zeros(s) for s in self.layer_sizes[1:]]
        # y is an integer, so we need to make a vector out of it first
        y_vector = np.zeros(self.layer_sizes[-1])
        y_vector[y] = 1
        # initialise the last layer delta errors
        delta[-1] = self.cost_function.cost_function_derivative(y_vector, a[-1]) * sigmoid_prime(z[-1])  # * is element-wise product
        # propagate back
        for l in range(self.number_of_layers-3,-1,-1):
            delta[l] = np.dot(np.transpose(self.weights[l+1]),delta[l+1]) * sigmoid_prime(z[l])   # * is element-wise product

        # calculate gradient of Cx
        nabla_weights_x = [np.zeros((size1,size2)) for size1,size2 in zip(self.layer_sizes[1:],self.layer_sizes[:-1])]
        nabla_biases_x = [np.zeros(s) for s in self.layer_sizes[1:]]
        l = 0
        for w,b in zip(self.weights,self.biases):
            if l == 0:
                # because x is basically the -1th activations (of input layer)
                nabla_weights_x[l] = np.outer(delta[l],x)
            else:
                nabla_weights_x[l] = np.outer(delta[l],a[l-1])
            
            nabla_biases_x[l] = delta[l]

            l += 1
        
        return nabla_weights_x, nabla_biases_x, correct
    
    #
    #   Classification Methods
    #
    
    # method to classify a single image given either as a 28x28 numpy array
    # or a 784-element 1D numpy array
    def classify(self, image):
        # flatten the numpy array in case it is a 28x28 array
        x = image.flatten()
        # take the max value position as the prediction
        return np.argmax(self.feed_forward(x))

    # method to classify a given list of images
    def classify_images(self, images):
        predictions = []
        for image in images:
            predictions.append(self.classify(image))
        return predictions
    
    #
    #   Parameter Initialisation Methods
    #
    
    # method to initialise the weights and biases of the network, but the weights
    # are sample from a normal distribution with corrected variance of 1/sqrt(k)
    # where k is the number of neurons in the previous layer.
    def initialise_variance_corrected_parameters(self):
        # sample biases from N(0,1)
        self.biases = [np.random.randn(s) for s in self.layer_sizes[1:]]
        # sample weights from N(0,1/(number of neurons in layer before))
        self.weights = [np.random.randn(x,y)/np.sqrt(y) for x,y in zip(self.layer_sizes[1:],self.layer_sizes[:-1])]

    # method to initialise the weights and biases of the network by sampling both
    # from the standard normal distribution
    def initialise_parameters(self):
        # sample biases from N(0,1)
        self.biases = [np.random.randn(s) for s in self.layer_sizes[1:]]
        # sample weights from N(0,1)
        self.weights = [np.random.randn(x,y) for x,y in zip(self.layer_sizes[1:],self.layer_sizes[:-1])]

    #
    #   Evaluation methods
    #
    
    # method to evaluate the NN model on the validation set.
    # returns the accuracy of the model on the validation set
    def evaluate_on_validation_set(self):
        nn_predictions = self.classify_images([img for img,_ in self.data.validation_set])
        ground_truth = [cat for _,cat in self.data.validation_set]

        correct_classifications = 0
        for i in range(len(self.data.validation_set)):
            if nn_predictions[i] == ground_truth[i]:
                correct_classifications += 1
        
        return correct_classifications / len(self.data.validation_set)
    
    # method to evaluate the NN model on the testing set.
    # returns the accuracy of the model on the testing set.
    # this method should be run only once to get the final
    # accuracy of the model, and avoid overfitting.
    def evaluate_on_test_set(self):
        nn_predictions = self.classify_images([img for img,_ in self.data.testing_set])
        ground_truth = [cat for _,cat in self.data.testing_set]

        correct_classifications = 0
        for i in range(len(self.data.testing_set)):
            if nn_predictions[i] == ground_truth[i]:
                correct_classifications += 1
        
        return correct_classifications / len(self.data.testing_set)