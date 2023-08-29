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
        
        # shuffle the larger dataset before dividing into training and validation sets
        # zipped = list(zip(x_train_and_val,y_train_and_val))
        # np.random.shuffle(zipped)
        # x_train_and_val = [x for x,_ in zipped]
        # y_train_and_val = [y for _,y in zipped]

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
    #
    # additionally, the MNIST images have pixel values from 0 to 255, so we first
    # normalise these values to the [0,1] range to be used by classifiers.
    def __init__(self, training_set, validation_set, testing_set, flatten_inputs = False):
        # normalise pixel values, and pair up inputs and ground truths
        training_set = [(x/255.0,y) for x,y in zip(training_set[0],training_set[1])]
        validation_set = [(x/255.0,y) for x,y in zip(validation_set[0],validation_set[1])]
        testing_set = [(x/255.0,y) for x,y in zip(testing_set[0],testing_set[1])]

        # flatten inputs if needed
        if flatten_inputs:
            training_set = [(x.flatten(),y) for x,y in training_set]
            validation_set = [(x.flatten(),y) for x,y in validation_set]
            testing_set = [(x.flatten(),y) for x,y in testing_set]
        
        self.training_set = training_set
        self.validation_set = validation_set
        self.testing_set = testing_set