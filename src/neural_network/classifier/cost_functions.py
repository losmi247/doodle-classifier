import numpy as np
from abc import ABC, abstractmethod

#
#   Utility Methods
#

# can be applied to a numpy array
def sigmoid(x):
    # clip the values to avoid overflow and vanishing gradients - removed for now
    # x = np.clip(x, -5, 5)
    return 1.0 / (1.0 + np.exp(-x))

# can be applied to a numpy array
def sigmoid_prime(x):
    y = sigmoid(x)
    return y * (1 - y)  # * is element-wise product

#
# Astract base class for a cost function used by the 
# Neural Network classifier. All cost functions must
# implement the abstract methods in this class
#
class CostFunction(ABC):
    # method to calculate the cost function on the train set
    @abstractmethod
    def calc_cost_function(neural_network):
        pass

    # method to calculate the derivative of the cost function C(x) by z(x)
    # for a single input x, where a(x) is the network's vector output for input x,
    # y(x) is the expected vector output for input x, and z(x) is the vector of
    # weighted inputs of output layer neurons (i.e. a(x) = sigmoid(z(x))).
    #
    # essentially, this method computes delta^L_{j} - i.e. derivatives of C(x) by
    # z^L_{j} (last layer weighted outputs).
    @abstractmethod
    def cost_function_derivative(y, a, z):
        pass

#
# Class for the mean squared error cost function:
#
#   C = 1/(2n) * (sum over inputs x in train set): ||y(x)-a(x)||^2
#     = 1/n * (sum over inputs x in train set): C(x)
#
# where y(x) is the expected vector output for input x, and
# a(x) is the network's vector output for input a(x), and n
# is the number of inputs in the train set.
#
class MeanSquaredError(CostFunction):
    # method to calculate the cost function on the train set
    @staticmethod
    def calc_cost_function(neural_network):
        c = 0
        for x,y in neural_network.data.training_set:
            a = neural_network.feed_forward(x)
            # y is an integer, so we need to make a vector out of it first
            y_vector = np.zeros(neural_network.layer_sizes[-1])
            y_vector[y] = 1

            # add C(x) to the cost function so far
            c += np.linalg.norm(y_vector-a)

        c /= 2*len(neural_network.data.training_set)

        return c
    
    # method to calculate the derivative of the cost function C(x) by z(x) 
    # for a single input x. Here:
    #
    #       C(x) = 0.5 * ||y(x)-a(x))||^2
    #
    # where y(x) is the correct vector output for input x, a(x) is the 
    # network's output for input x, and z(x) is the vector of weighted inputs
    # of output layer neurons (i.e. a(x) = sigmoid(z(x))).
    #
    # essentially, this method computes delta^L_{j} - i.e. derivatives of C(x) by
    # z^L_{j} (last layer weighted outputs).
    @staticmethod
    def cost_function_derivative(y, a, z):
        return (a - y) * sigmoid_prime(z)
    

#
# Class for the cross entropy cost function:
#
#   C = -(1/n) * (sum over inputs x in train set): (sum over neurons j in last layer): y(x)_j*ln(a(x)_j) + (1−y(x)_j)*ln(1−a(x)_j)
#     = -(1/n) * (sum over inputs x in train set): C(x)
#
# where y(x) is the expected vector output for input x, and
# a(x) is the network's vector output for input a(x), and n
# is the number of inputs in the train set.
#
class CrossEntropy(CostFunction):
    # method to calculate the cost function on the train set
    @staticmethod
    def calc_cost_function(neural_network):
        c = 0
        for x,y in neural_network.data.training_set:
            a = neural_network.feed_forward(x)
            # y is an integer, so we need to make a vector out of it first
            y_vector = np.zeros(neural_network.layer_sizes[-1])
            y_vector[y] = 1
            
            for h in a:
                assert(h >= 0 and h <= 1)

            # subtract C(x) from the cost function so far
            c -= np.sum(np.nan_to_num(y*np.log(a) + (1-y)*np.log(1-a)))

        c /= len(neural_network.data.training_set)

        return c
    
    # method to calculate the derivative of the cost function C(x) by z(x)
    # for a single input x. Here:
    #
    #       C(x) = - (sum over output neuron j): (y(x)_j*ln(a(x)_j) + (1−y(x)_j)*ln(1−a(x)_j))
    #
    # where y(x) is the correct vector output for input x, a(x) is the 
    # network's output for input x, and z(x) is the vector of weighted inputs
    # of output layer neurons (a(x) = sigmoid(z(x))). this method does not use the output 
    # layer's weighted inputs, so cross entropy's gradients are larger.
    #
    # essentially, this method computes delta^L_{j} - i.e. derivatives of C(x) by
    # z^L_{j} (last layer weighted outputs).
    @staticmethod
    def cost_function_derivative(y, a, z):
        return (a - y)