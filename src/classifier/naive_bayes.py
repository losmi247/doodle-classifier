import numpy as np
import math
from classifier.features import *

#
# Class for a Naive Bayes classifier.
#
class NaiveBayesClassifier:
    features = NBFeature.get_all_features()

    def __init__(self, categories, data):
        self.categories = categories

        self.log_priors = {}
        self.log_likelihoods = {}

        self.data = data

        self.trained = False

    # method to train the NB model, i.e. estimate the parameters
    def train(self):
        self.estimate_priors()
        self.estimate_likelihoods()
        self.trained = True

    # method to estimate priors
    def estimate_priors(self):
        category_cnt = {}

        # add-1 smoothing
        for category in self.categories:
            category_cnt[category] = 1

        # count categories
        for category in self.data.training_set[1]:
            category_cnt[category] += 1

        # maximum likelihood estimate
        for category,count in category_cnt.items():
            self.log_priors[category] = math.log(count / self.categories.size)
        
    # method to estimate likelihoods
    def estimate_likelihoods(self):
        feature_category_cnt = {}
        feature_cnt = {}

        # add-1 smoothing
        for feature in NaiveBayesClassifier.features:
            for category in self.categories:
                feature_category_cnt[feature, category] = 1

        # extract images of each type
        images_of_category = {}
        for category in self.categories:
            images_of_category[category] = []
        for image,cat in zip(self.data.training_set[0], self.data.training_set[1]):
            images_of_category[cat].append(image)
        for category in self.categories:
            images_of_category[category] = np.array(images_of_category[category])

        # count features in different categories
        for category in self.categories:
            # count features in all images of this category
            feature_frequency = NBFeature.count_features_in_images(images_of_category[category])
            
            # count total number of features in the selected images
            feature_cnt[category] = sum(feature_frequency.values())

            for feature,count in feature_frequency.items():
                feature_category_cnt[feature, category] += count
            
            print("Training progress: ", (category+1)*10, "%")
        
        # maximum likelihood estimate
        for (feature,category),count in feature_category_cnt.items():
            self.log_likelihoods[feature, category] = \
                math.log(count / (feature_cnt[category] + NaiveBayesClassifier.features.size))

    # method to classify a given 28x28 pixels image
    def classify(self, image):
        if not self.trained:
            raise Exception("Classify method invoked before training the NB model.")
        
        # extract all known features from the image
        features = NBFeature.extract_features(image)

        max_probability = -1000000000
        prediction = -1
        for category in self.categories:
            log_probability = self.log_priors[category]
            
            for feature in features:
                if (feature,category) not in self.log_likelihoods:
                    raise Exception("Feature not present in log-likelihoods.")
                
                log_probability += self.log_likelihoods[feature, category]
            
            if log_probability > max_probability:
                max_probability = log_probability
                prediction = category

        return prediction
    
    # method to classify a given list of 28x28 pixels images
    def classify_images(self, images):
        predictions = []
        for image in images:
            predictions.append(self.classify(image))
        return predictions
    
    # method to evaluate the NB model on the testing set,
    # returns the accuracy of the model on the test set
    def evaluate(self):
        nb_predictions = self.classify_images(self.data.testing_set[0])
        ground_truth = self.data.testing_set[1]

        correct_classifications = 0
        for i in range(len(self.data.testing_set[0])):
            if nb_predictions[i] == ground_truth[i]:
                correct_classifications += 1
        
        return correct_classifications / len(self.data.testing_set[0])
