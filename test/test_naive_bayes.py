import unittest
import numpy as np
import matplotlib.pyplot as plt
from os.path import join

from src.naive_bayes.classifier.naive_bayes import NaiveBayesClassifier
from src.utility.loader import *

input_path = './data'
training_images_filepath = join(input_path, 'train-images-idx3-ubyte/train-images-idx3-ubyte')
training_labels_filepath = join(input_path, 'train-labels-idx1-ubyte/train-labels-idx1-ubyte')
test_images_filepath = join(input_path, 't10k-images-idx3-ubyte/t10k-images-idx3-ubyte')
test_labels_filepath = join(input_path, 't10k-labels-idx1-ubyte/t10k-labels-idx1-ubyte')

class Testing(unittest.TestCase):
    # NaiveBayesClassifier constructor
    def test_nb_constructor(self):
        nb = NaiveBayesClassifier(np.arange(9), None)
        
        i = 0
        for x in nb.categories:
            self.assertEqual(x,i)
            i += 1
            
    # train
    def test_training(self):
        # load data
        mnist_dataloader = MnistDataloader(training_images_filepath, training_labels_filepath, test_images_filepath, test_labels_filepath)
        # (images, labels)
        (x_train, y_train), (x_validation, y_validation), (x_test, y_test) = mnist_dataloader.load_data()
        nbdata = Data((x_train, y_train),(x_validation, y_validation),(x_test, y_test))

        # create a NB classifier and train it
        nb_classifier = NaiveBayesClassifier(np.arange(10), nbdata)
        nb_classifier.train()
        
        self.assertTrue(nb_classifier.evaluate_on_validation_set() >= 0.83)
        
    # save_nb, load_nb
    def test_save_and_load_nb(self):
        # load data
        mnist_dataloader = MnistDataloader(training_images_filepath, training_labels_filepath, test_images_filepath, test_labels_filepath)
        # (images, labels)
        (x_train, y_train), (x_validation, y_validation), (x_test, y_test) = mnist_dataloader.load_data()
        nbdata = Data((x_train, y_train),(x_validation, y_validation),(x_test, y_test))
        
        # create a NB classifier and train it
        nb_classifier = NaiveBayesClassifier(np.arange(10), nbdata)
        nb_classifier.train()
        
        # save the NB classifier
        nb_classifier.save_nb("test_save_nb.txt")
        
        # load the NB classifier, get the deployable model
        deployable_nb = NaiveBayesClassifier.load_nb("test_save_nb.txt")
        
        # assert that the priors remain the same
        self.assertTrue(all([nb_classifier.log_priors.get(k) == v for k,v in deployable_nb.log_priors.items()]))
        # assert that the likelihoods remain the same
        self.assertTrue(all([nb_classifier.log_likelihoods[feat,cat] == v for (feat,cat),v in deployable_nb.log_likelihoods.items()]))
        
    # save_nb, load_nb
    def test_save_and_load_nb_accuracy(self):
        # load data
        mnist_dataloader = MnistDataloader(training_images_filepath, training_labels_filepath, test_images_filepath, test_labels_filepath)
        # (images, labels)
        (x_train, y_train), (x_validation, y_validation), (x_test, y_test) = mnist_dataloader.load_data()
        # for NB, we don't flatten inputs (they remain 28x28), and we don't normalise inputs (they remain in range 0-255)
        nbdata = Data((x_train, y_train),(x_validation, y_validation),(x_test, y_test))

        # load the NB, get the deployable model
        deployable_nb = NaiveBayesClassifier.load_nb("bayes.txt")

        # assert that the accuracy on training and validation sets are at least 80%
        self.assertTrue(deployable_nb.evaluate_on_dataset(nbdata.training_set) >= 0.83)
        self.assertTrue(deployable_nb.evaluate_on_dataset(nbdata.validation_set) >= 0.83)
            

if __name__ == '__main__':
    unittest.main()