# script to run the naive bayes classifier - use
#
#   python3 -m src.naive_bayes.run
#
# to run it (from project root directory).
#
# partly from https://www.kaggle.com/code/milospuric/read-mnist-dataset/edit
from src.utility.loader import *
from src.naive_bayes.classifier.features import *
from src.naive_bayes.classifier.naive_bayes import *
from os.path import join
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
    nbdata = Data((x_train, y_train),(x_validation, y_validation),(x_test, y_test))

    # create a NB classifier and train it
    nb_classifier = NaiveBayesClassifier(np.arange(10), nbdata)
    nb_classifier.train()

    # achieves 84% validation accuracy
    print("Accuracy on validation set: ", nb_classifier.evaluate_on_validation_set())
    
    # save the trained model's parameters
    nb_classifier.save_nb("bayes.txt")
    
    ind = 101
    img = x_validation[ind]
    pred = (DeployableNaiveBayes(nb_classifier.log_priors, nb_classifier.log_likelihoods, nb_classifier.categories).classify(img))
    print("Predicted: ", pred, ". True: ", y_validation[ind])
    plt.imshow(img, cmap=plt.cm.gray)
    plt.show()

if __name__ == "__main__":
    main()
