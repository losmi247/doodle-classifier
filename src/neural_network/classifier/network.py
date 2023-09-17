import numpy as np
import random
import pickle

from src.neural_network.classifier.cost_functions import *

#
#   Utility Methods
#

# can be applied to a numpy array
def sigmoid(x):
    # clip the values to avoid overflow and vanishing gradients - removed for now
    x = np.clip(x, -200, 200)
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
    def __init__(self, layer_sizes, data, cost_function = MeanSquaredError):
        self.number_of_layers = len(layer_sizes)
        self.layer_sizes = layer_sizes

        # initialise parameters of the network - weights and biases
        self.initialise_variance_corrected_parameters()

        # the data
        self.data = data

        # set the cost function used
        self.cost_function = cost_function
        
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
    
    # method to train the neural network - default arguments achieve 74% validation accuracy
    def train(self, epochs = 30, m = 50, learning_rate = 0.01):
        accuracies = []
        cost_functions = []

        for i in range(epochs):
            # shuffle the training data before taking mini batches - removed for now
            random.shuffle(self.data.training_set)
            # make the mini batches - the last mini batch might be smaller than the rest if len(self.data.training_set)%m != 0
            mini_batches = [self.data.training_set[k*m:(k+1)*m] for k in range((len(self.data.training_set)+m-1)//m)]

            for mini_batch in mini_batches:
                self.process_mini_batch(mini_batch, learning_rate)

            print("Epoch ", i+1, "/", epochs, " finished.", end=" ")

            training_accuracy = self.accuracy_on_train_set()
            print("Accuracy on training set: ", training_accuracy, end=", ")
            accuracies.append(training_accuracy)

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
        
        # go through each input in the mini batch and find gradient of Cx using backpropagation
        for x,y in mini_batch:
            nabla_weights_x, nabla_biases_x = self.backpropagate(x,y)
    
            # sum up the contributions
            nabla_weights = [nw+nwx for nw,nwx in zip(nabla_weights,nabla_weights_x)]
            nabla_biases = [nb+nbx for nb,nbx in zip(nabla_biases,nabla_biases_x)]
        
        # move by -n*gradient(C) where C is the cost function - gradient descent
        self.weights = [w-(learning_rate/len(mini_batch))*nw for w,nw in zip(self.weights,nabla_weights)]
        self.biases = [b-(learning_rate/len(mini_batch))*nb for b,nb in zip(self.biases,nabla_biases)]
        
    # method to calculate the gradient of the cost function C(x) for one input x
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
                z[l] = np.dot(w,a[l-1])+b
            a[l] = sigmoid(z[l])
            l += 1

        # the derivatives by z
        delta = [np.zeros(s) for s in self.layer_sizes[1:]]
        # y is an integer, so we need to make a vector out of it first
        y_vector = np.zeros(self.layer_sizes[-1])
        y_vector[y] = 1
        # initialise the last layer delta errors
        delta[-1] = self.cost_function.cost_function_derivative(y_vector, a[-1], z[-1])
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
        
        return nabla_weights_x, nabla_biases_x
    
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
    #   Evaluation methods (all have to create deployable models first)
    #

    # method to calculate the accuracy of the model on the training set
    def accuracy_on_train_set(self):
        model = DeployableNetwork(self.weights, self.biases)
        nn_predictions = model.classify_images([img for img,_ in self.data.training_set])
        ground_truth = np.array([cat for _,cat in self.data.training_set])
        
        return np.sum(nn_predictions == ground_truth) / len(self.data.training_set)
    
    # method to evaluate the NN model on the validation set.
    # returns the accuracy of the model on the validation set
    def evaluate_on_validation_set(self):
        model = DeployableNetwork(self.weights, self.biases)
        nn_predictions = model.classify_images([img for img,_ in self.data.validation_set])
        ground_truth = np.array([cat for _,cat in self.data.validation_set])

        return np.sum(nn_predictions == ground_truth) / len(self.data.validation_set)
    
    # method to evaluate the NN model on the testing set.
    # returns the accuracy of the model on the testing set.
    # this method should be run only once to get the final
    # accuracy of the modeland avoid overfitting.
    def evaluate_on_test_set(self):
        model = DeployableNetwork(self.weights, self.biases)
        nn_predictions = model.classify_images([img for img,_ in self.data.testing_set])
        ground_truth = np.array([cat for _,cat in self.data.testing_set])

        return np.sum(nn_predictions == ground_truth) / len(self.data.testing_set)
    
    #
    #   Utility Methods
    #
    
    # method to save the weights and biases of a trained neural network in a file
    # with the given name in this classifier's 'trained_models' directory.
    # only the weights and biases are saved.
    def save_network(self, file_name):
        with open("./src/neural_network/trained_models/"+file_name, "wb") as file:
            file.write(self.pickle_network())
        
    # method to serialise the network parameters
    def pickle_network(self):
        return pickle.dumps(
            [pickle.dumps(w) for w in self.weights] + [pickle.dumps(b) for b in self.biases]
        )
    
    # method to load the weights and biases from a file with the given name
    # in this classifiers's 'trained_models' directory, and create a new
    # deployable neural network (see class DeployableNetwork) with those 
    # parameters, and returns it
    @staticmethod
    def load_network(file_name):
        with open("./src/neural_network/trained_models/"+file_name, "rb") as file:
            # first unpickle the array of matrices and vectors
            pickled_params = pickle.loads(file.read())
            # then unpickle individual matrices and vectors
            weights = [pickle.loads(w_pickled) for w_pickled in pickled_params[:len(pickled_params)//2]]
            biases = [pickle.loads(b_pickled) for b_pickled in pickled_params[len(pickled_params)//2:]]
            # create and return the deployable model
            return DeployableNetwork(weights, biases)


#
#   Class for a deployable neural network, consisting of only
#   weights and biases, without any data, cost functions or
#   layer sizes.
#
class DeployableNetwork:
    def __init__(self, weights, biases):
        self.weights = weights
        self.biases = biases
        
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
    #   Classification Methods - the images must be given with pixels
    #   in range [0,1] (normalised)
    #
    
    # method to classify a single image given either as a 28x28 numpy array
    # or a 784-element 1D numpy array
    def classify(self, image):
        # flatten the numpy array in case it is a 28x28 array, and 
        # take the max value position as the prediction
        return np.argmax(self.feed_forward(image.flatten()))

    # method to classify a given list of images
    def classify_images(self, images):
        return np.array([self.classify(image) for image in images])
    
    
    #
    #   Evaluation Methods 
    #

    # method to evaluate the deployable NN model on the given dataset.
    # returns the accuracy of the model on the given dataset.
    def evaluate_on_dataset(self, dataset):
        nn_predictions = self.classify_images([img for img,_ in dataset])
        ground_truth = np.array([cat for _,cat in dataset])

        return np.sum(nn_predictions == ground_truth) / len(dataset)