# script to run the naive bayes classifier - use
#
#   python3 -m src.neural_network.run
#
# to run it (from project root directory).
#
# partly from https://www.kaggle.com/code/milospuric/read-mnist-dataset/edit
from src.utility.loader import *
from src.neural_network.classifier.network import *
from os.path import join
import numpy as np
import matplotlib.pyplot as plt 

#
# Set file paths based on added MNIST Datasets
#
input_path = './data'
training_images_filepath = join(input_path, 'train-images-idx3-ubyte/train-images-idx3-ubyte')
training_labels_filepath = join(input_path, 'train-labels-idx1-ubyte/train-labels-idx1-ubyte')
test_images_filepath = join(input_path, 't10k-images-idx3-ubyte/t10k-images-idx3-ubyte')
test_labels_filepath = join(input_path, 't10k-labels-idx1-ubyte/t10k-labels-idx1-ubyte')

def main():
    # load data
    mnist_dataloader = MnistDataloader(training_images_filepath, training_labels_filepath, test_images_filepath, test_labels_filepath)
    # (images, labels)
    (x_train, y_train), (x_validation, y_validation), (x_test, y_test) = mnist_dataloader.load_data()
    # for NNs, we need to flatten the 28x28 image to a 1D numpy array of length 784
    nndata = Data((x_train, y_train),(x_validation, y_validation),(x_test, y_test), flatten_inputs=True)

    # create a NN and train it - this setup gave 39% accuracy on validation set
    nn = NeuralNetwork([784,30,10],nndata)
    number_of_epochs = 30
    mini_batch_size = 10
    eta = 0.07
    accuracies, cost_functions = nn.train(epochs=number_of_epochs, m=mini_batch_size, learning_rate=eta)

    # evaluate the model on the validation set
    print("Accuracy on validation set: ", nn.evaluate_on_validation_set())
    
    # plot stuff
    fig, (ax1, ax2) = plt.subplots(1, 2)

    # plot the accuracy and cost function after each epoch
    ax1.plot(np.arange(number_of_epochs), accuracies, color='blue', label='accuracy on train set after each epoch')
    ax1.plot(np.arange(number_of_epochs), cost_functions, color='red', label='cost function after each epoch')
    ax1.legend()

    # show an example from the validation set
    ind = 2543
    img = x_validation[ind]
    pred = nn.classify(img)
    print("Predicted: ", pred, ". True: ", y_validation[ind])
    ax2.imshow(img, cmap=plt.cm.gray)

    plt.show()

if __name__ == "__main__":
    main()
