import numpy as np
import struct
from array import array

#
# MNIST Data Loader Class from https://www.kaggle.com/code/milospuric/read-mnist-dataset/edit.
#
class MnistDataloader(object):
    def __init__(self, training_images_filepath,training_labels_filepath,
                 test_images_filepath, test_labels_filepath):
        self.training_images_filepath = training_images_filepath
        self.training_labels_filepath = training_labels_filepath
        self.test_images_filepath = test_images_filepath
        self.test_labels_filepath = test_labels_filepath
    
    def read_images_labels(self, images_filepath, labels_filepath):        
        labels = []
        with open(labels_filepath, 'rb') as file:
            magic, size = struct.unpack(">II", file.read(8))
            if magic != 2049:
                raise ValueError('Magic number mismatch, expected 2049, got {}'.format(magic))
            labels = array("B", file.read())        
        
        with open(images_filepath, 'rb') as file:
            magic, size, rows, cols = struct.unpack(">IIII", file.read(16))
            if magic != 2051:
                raise ValueError('Magic number mismatch, expected 2051, got {}'.format(magic))
            image_data = array("B", file.read())        
        images = []
        for i in range(size):
            images.append([0] * rows * cols)
        for i in range(size):
            img = np.array(image_data[i * rows * cols:(i + 1) * rows * cols])
            img = img.reshape(28, 28)
            images[i] = img   
        
        return images, labels
            
    def load_data(self):
        x_train_and_val, y_train_and_val = self.read_images_labels(self.training_images_filepath, self.training_labels_filepath)
        x_test, y_test = self.read_images_labels(self.test_images_filepath, self.test_labels_filepath)

        x_train, y_train = x_train_and_val[:50000], y_train_and_val[:50000]
        x_validation, y_validation = x_train_and_val[50000:], y_train_and_val[50000:]

        return (x_train, y_train),(x_validation, y_validation),(x_test, y_test)


#
# Class for the data that the classifiers use - 70000 28x28 pixels images, 
# 50000 for training, 10000 for validation (tuning parameters, and 10000 for testing.
#
class Data:
    # constructor from the parsed train, validation, and test sets.
    # originally, each input is a 28x28 numpy array, and a flag 'flatten_inputs'
    # can be activated to flatten this into a 1D numpy array of length 784 - for
    # instance that is what neural networks need (row major order for flattening).
    def __init__(self, training_set, validation_set, testing_set, flatten_inputs = False):
        if flatten_inputs:
            training_set = [(x.flatten(),y) for x,y in zip(training_set[0],training_set[1])]
            validation_set = [(x.flatten(),y) for x,y in zip(validation_set[0],validation_set[1])]
            testing_set = [(x.flatten(),y) for x,y in zip(testing_set[0],testing_set[1])]
        else:
            training_set = list(zip(training_set[0], training_set[1]))
            validation_set = list(zip(validation_set[0], validation_set[1]))
            testing_set = list(zip(testing_set[0], testing_set[1]))
        
        self.training_set = training_set
        self.validation_set = validation_set
        self.testing_set = testing_set