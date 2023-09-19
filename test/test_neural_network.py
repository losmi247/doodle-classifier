import unittest
import numpy as np
import matplotlib.pyplot as plt
from os.path import join

from src.neural_network.classifier.network import NeuralNetwork
from src.utility.loader import *
from src.neural_network.classifier.cost_functions import *

input_path = './data'
training_images_filepath = join(input_path, 'train-images-idx3-ubyte/train-images-idx3-ubyte')
training_labels_filepath = join(input_path, 'train-labels-idx1-ubyte/train-labels-idx1-ubyte')
test_images_filepath = join(input_path, 't10k-images-idx3-ubyte/t10k-images-idx3-ubyte')
test_labels_filepath = join(input_path, 't10k-labels-idx1-ubyte/t10k-labels-idx1-ubyte')

class Testing(unittest.TestCase):
    # NeuralNetwork constructor
    def test_network_constructor(self):
        network = NeuralNetwork([784,30,10],None,categories=np.arange(10),cost_function=MeanSquaredError,lmbda=1)
        
        x = network.weights[0]
        self.assertEquals(x.shape[0], 30)
        self.assertEquals(x.shape[1], 784)
        
        x = network.weights[1]
        self.assertEquals(x.shape[0], 10)
        self.assertEquals(x.shape[1], 30)
        
        x = network.biases[0]
        self.assertEquals(x.shape[0], 30)
        
        x = network.biases[1]
        self.assertEquals(x.shape[0], 10)
        
        
        x = network.categories[6]
        self.assertEquals(x, 6)
        
        self.assertEquals(network.lmbda, 1)
    
    # train
    def test_train(self):
        # load data
        mnist_dataloader = MnistDataloader(training_images_filepath, training_labels_filepath, test_images_filepath, test_labels_filepath)
        # (images, labels)
        (x_train, y_train), (x_validation, y_validation), (x_test, y_test) = mnist_dataloader.load_data()
        # for NNs, we need to flatten the 28x28 image to a 1D numpy array of length 784
        nndata = Data((x_train, y_train),(x_validation, y_validation),(x_test, y_test), normalise_inputs=True, flatten_inputs=True)

        # create a NN and train it
        nn = NeuralNetwork([784,30,10], nndata, categories=np.arange(10), cost_function=CrossEntropy, lmbda=0.1)
        number_of_epochs = 5
        mini_batch_size = 50
        eta = 0.01
        accuracies, cost_functions = nn.train(epochs=number_of_epochs, m=mini_batch_size, learning_rate=eta)
        
        # assert that accuracies are increasing
        self.assertTrue((np.diff(accuracies) > 0).all())
        # assert that the cost function is decreasing
        self.assertTrue((np.diff(cost_functions) < 0).all())

        # evaluate the model on the validation set
        print("Accuracy on validation set: ", nn.evaluate_on_validation_set())
        
        # plot stuff
        fig, (ax1, ax2) = plt.subplots(1, 2)

        # plot the accuracy and cost function after each epoch
        ax1.plot(np.arange(number_of_epochs), accuracies, color='blue', label='accuracy on train set after each epoch')
        ax1.plot(np.arange(number_of_epochs), cost_functions, color='red', label='cost function after each epoch')
        ax1.legend()
        
    # save_network, load_network
    def test_save_and_load_network(self):
        # load data
        mnist_dataloader = MnistDataloader(training_images_filepath, training_labels_filepath, test_images_filepath, test_labels_filepath)
        # (images, labels)
        (x_train, y_train), (x_validation, y_validation), (x_test, y_test) = mnist_dataloader.load_data()
        # for NNs, we need to flatten the 28x28 image to a 1D numpy array of length 784
        nndata = Data((x_train, y_train),(x_validation, y_validation),(x_test, y_test), normalise_inputs=True, flatten_inputs=True)
        
        nn = NeuralNetwork([784,30,10], nndata, categories=np.arange(10), cost_function=CrossEntropy, lmbda=0.1)
        number_of_epochs = 5
        mini_batch_size = 50
        eta = 0.01
        accuracies, cost_functions = nn.train(epochs=number_of_epochs, m=mini_batch_size, learning_rate=eta)
        
        # evaluate the model on the validation set
        print("Accuracy on validation set: ", nn.evaluate_on_validation_set())
        
        # save the network
        nn.save_network("test_save_network.txt")
        
        # load the network, get the deployable network
        deployable_network = NeuralNetwork.load_network("test_save_network.txt")
        
        # assert that the weights remain the same
        for w_dep,w_orig in zip(deployable_network.weights, nn.weights):
            self.assertTrue((w_dep==w_orig).all())
            
        # assert that the biases remain the same
        for b_dep,b_orig in zip(deployable_network.biases, nn.biases):
            self.assertTrue((b_dep==b_orig).all())

    # save_network, load_network
    def test_save_and_load_network_accuracy(self):
        # load data
        mnist_dataloader = MnistDataloader(training_images_filepath, training_labels_filepath, test_images_filepath, test_labels_filepath)
        # (images, labels)
        (x_train, y_train), (x_validation, y_validation), (x_test, y_test) = mnist_dataloader.load_data()
        # for NNs, we need to flatten the 28x28 image to a 1D numpy array of length 784
        nndata = Data((x_train, y_train),(x_validation, y_validation),(x_test, y_test), normalise_inputs=True, flatten_inputs=True)

        # load the network, get the deployable network
        deployable_network = NeuralNetwork.load_network("network.txt")

        # assert that the accuracy on training and validation sets are at least 90%
        self.assertTrue(deployable_network.evaluate_on_dataset(nndata.training_set) >= 0.93)
        self.assertTrue(deployable_network.evaluate_on_dataset(nndata.validation_set) >= 0.93)
        

if __name__ == '__main__':
    unittest.main()