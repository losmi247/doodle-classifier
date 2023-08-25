# script to run the naive bayes classifier, 
# parts from https://www.kaggle.com/code/milospuric/read-mnist-dataset/edit
from utility.loader import *
from classifier.features import *
from classifier.naive_bayes import *
from os.path import join
import matplotlib.pyplot as plt

#
# Set file paths based on added MNIST Datasets
#
input_path = './data/archive'
training_images_filepath = join(input_path, 'train-images-idx3-ubyte/train-images-idx3-ubyte')
training_labels_filepath = join(input_path, 'train-labels-idx1-ubyte/train-labels-idx1-ubyte')
test_images_filepath = join(input_path, 't10k-images-idx3-ubyte/t10k-images-idx3-ubyte')
test_labels_filepath = join(input_path, 't10k-labels-idx1-ubyte/t10k-labels-idx1-ubyte')

def main():
    # load data
    mnist_dataloader = MnistDataloader(training_images_filepath, training_labels_filepath, test_images_filepath, test_labels_filepath)
    # (images, labels)
    (x_train, y_train), (x_test, y_test) = mnist_dataloader.load_data()
    nbdata = NBData((x_train, y_train), (x_test, y_test))

    # create a NB classifier and train it
    nb_classifier = NaiveBayesClassifier(np.arange(10), nbdata)
    nb_classifier.train()

    print(nb_classifier.evaluate())
    
    ind = 101
    img = x_test[ind]
    pred = nb_classifier.classify(img)
    print("Predicted: ", pred, ". True: ", y_test[ind])
    plt.imshow(img, cmap=plt.cm.gray)
    plt.show()

if __name__ == "__main__":
    main()
