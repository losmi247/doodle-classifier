import numpy as np
from abc import ABC, abstractmethod

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

    # method to calculate the derivative of the cost function C(x) for a single
    # input x by a(x), where a(x) is the network's vector output for input x,
    # and y(x) is the expected vector output for input x.
    @abstractmethod
    def cost_function_derivative(y, a):
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
    def calc_cost_function(neural_network):
        c = 0
        for x,y in neural_network.data.training_set:
            a = neural_network.feed_forward(x)
            # y is an integer, so we need to make a vector out of it first
            y_vector = np.zeros(neural_network.layer_sizes[-1])
            y_vector[y] = 1
            c += np.dot(a-y_vector,a-y_vector)

        c /= 2*len(neural_network.data.training_set)

        return c
    
    # method to calculate the derivative of the cost function C(x) for a single
    # input x by a(x). Here:
    #       C(x) = 0.5 * ||y(x)-a(x))||^2 
    # where y(x) is the correct vector output for input x, and a(x) is the 
    # network's output for input x.
    def cost_function_derivative(y, a):
        return (a - y)