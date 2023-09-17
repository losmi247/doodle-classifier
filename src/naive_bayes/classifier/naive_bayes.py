import numpy as np
import math
import pickle

from src.naive_bayes.classifier.features import *

#
# Class for a Naive Bayes classifier.
#
class NaiveBayesClassifier:
    # numpy array of numpy arrays - each internal numpy array
    # contains first the type of the features in this array and
    # then all possible existing features of that type.
    features = NBFeature.get_all_features()
    
    # a dictionary of dictionaries. maps each implementation of NBFeature
    # (i.e. one for each feature type) to another dictionary. each internal dictionary 
    # maps every feature's (of that type) code (obtained by encode_feature(feature) function
    # in that feature's class)
    feature_decoder = {}
    for feature_type_list in features:
            feature_codes = {}
            feature_type_class = feature_type_list[0]
            for feature in feature_type_list[1:]:
                # map this feature's code to itself
                feature_codes[feature_type_class.encode_feature(feature)] = feature
            # map this feature type to the dictionary of codes to features
            feature_decoder[feature_type_class] = feature_codes

    def __init__(self, categories, data):
        self.categories = categories

        self.log_priors = {}
        self.log_likelihoods = {}

        self.data = data

        self.trained = False

    
    #
    #   Training Methods
    #

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
        for _,category in self.data.training_set:
            category_cnt[category] += 1

        # maximum likelihood estimate
        for category,count in category_cnt.items():
            self.log_priors[category] = math.log(count / self.categories.size)
        
    # method to estimate likelihoods
    def estimate_likelihoods(self):
        feature_category_cnt = {}
        feature_cnt = {}

        # add-1 smoothing
        for feature_type_list in NaiveBayesClassifier.features:
            for feature in feature_type_list[1:]:
                for category in self.categories:
                    feature_category_cnt[feature, category] = 1

        # extract images of each type
        images_of_category = {}
        for category in self.categories:
            images_of_category[category] = []
        for image,cat in self.data.training_set:
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
                math.log(count / (feature_cnt[category] + np.sum([feature_type_list.size-1 for feature_type_list in NaiveBayesClassifier.features])))
            

    #
    #   Evaluation Methods (all have to create deployable models first)
    #

    # method to evaluate the NB model on the training set.
    # returns the accuracy of the model on the training set
    def evaluate_on_validation_set(self):
        model = DeployableNaiveBayes(self.log_priors, self.log_likelihoods, self.categories)
        nb_predictions = model.classify_images([img for img,_ in self.data.training_set])
        ground_truth = np.array([cat for _,cat in self.data.training_set])
    
        return np.sum(nb_predictions == ground_truth) / len(self.data.training_set)
    
    # method to evaluate the NB model on the validation set.
    # returns the accuracy of the model on the validation set
    def evaluate_on_validation_set(self):
        model = DeployableNaiveBayes(self.log_priors, self.log_likelihoods, self.categories)
        nb_predictions = model.classify_images([img for img,_ in self.data.validation_set])
        ground_truth = np.array([cat for _,cat in self.data.validation_set])
    
        return np.sum(nb_predictions == ground_truth) / len(self.data.validation_set)

    # method to evaluate the NB model on the testing set.
    # returns the accuracy of the model on the testing set.
    # this method should be run only once to get the final
    # accuracy of the model, and avoid overfitting.
    def evaluate_on_test_set(self):
        model = DeployableNaiveBayes(self.log_priors, self.log_likelihoods, self.categories)
        nb_predictions = model.classify_images([img for img,_ in self.data.testing_set])
        ground_truth = np.array([cat for _,cat in self.data.testing_set])
    
        return np.sum(nb_predictions == ground_truth) / len(self.data.testing_set)
    

    #
    #   Utility Methods
    #

    # method to save the paremeters of a trained Naive Bayes classifier
    # in a file with the given name in this classifier's 'trained_models' 
    # directory. only the priors, likelihoods, and categories are saved.
    def save_nb(self, file_name):
        with open("./src/naive_bayes/trained_models/"+file_name, "wb") as file:
            file.write(self.pickle_nb())
        
    # method to serialise the Naive Bayes parameters
    def pickle_nb(self):
        return pickle.dumps((pickle.dumps(self.log_priors),pickle.dumps(self.log_likelihoods),pickle.dumps(self.categories)))
    
    # method to load the parameters from a file with the given name
    # in this classifiers's 'trained_models' directory, and create a new
    # deployable Naive Bayes classifier (see class DeployableNaiveBayes) 
    # with those parameters, and returns it
    @staticmethod
    def load_nb(file_name):
        with open("./src/naive_bayes/trained_models/"+file_name, "rb") as file:
            # first unpickle the parameters
            pickled_params = pickle.loads(file.read())
            
            # unpickle priors
            log_priors = pickle.loads(pickled_params[0])
            # unpickle categories
            categories = pickle.loads(pickled_params[2])
            
            # unpickle likelihoods - this will create new objects for each feature
            log_likelihoods_with_new_feature_objects = pickle.loads(pickled_params[1])
            log_likelihoods = {}
            for (new_feature_object,category),value in log_likelihoods_with_new_feature_objects.items():
                # find the type of this feature
                feature_type_class = None
                for feature_class in NBFeature.all_feature_classes:
                    if isinstance(new_feature_object, feature_class):
                        feature_type_class = feature_class
                
                # find the unique code for this combination of field value that this new feature object has
                unique_feature_code = feature_type_class.encode_feature(new_feature_object)
                # find the old feature object (the one originally created in the implementation of NBFeature)
                # using the decoder that maps feature codes to feature objects.
                old_feature = NaiveBayesClassifier.feature_decoder[feature_type_class][unique_feature_code]
                
                # use the old feature instead of the new one
                log_likelihoods[old_feature, category] = value
            
            """
            # update the 'all_features' static variable of each feature class so that
            # it contains the new objects that have been created by unpickling log_likelihoods.
            for (new_feature_object,category),_ in log_likelihoods.items():
                # each feature will be associated with each category, so we just go through one category
                if category != categories[0]:
                    continue
                
                # find which type of feature it is
                feature_type_class = None
                for feature_class in NBFeature.all_feature_classes:
                    if isinstance(new_feature_object, feature_class):
                        feature_type_class = feature_class
                
                # find the index of this feature in its class's 'all_features' array
                feature_index = feature_type_class.encode_feature(new_feature_object)
                # update the 'all_features' array with the new feature object
                feature_type_class.all_features[feature_index] = new_feature_object
            """
 
            # create and return the deployable model
            return DeployableNaiveBayes(log_priors, log_likelihoods, categories)
    

#
#   Class for a deployable NB classifier, consisting of only
#   priors, likelihoods and categories, without any data or features.
#
class DeployableNaiveBayes:
    def __init__(self, log_priors, log_likelihoods, categories):
        self.log_priors = log_priors
        self.log_likelihoods = log_likelihoods
        self.categories = categories

    
    #
    #   Methods
    #
    
    # method to get the probabilities for each class
    def get_probabilities(self, image):
        # we already have all parameters
        
        # extract all known existing features from the image
        image_features = NBFeature.extract_features(image)

        probabilities = []
        for category in self.categories:
            log_probability = self.log_priors[category]
            
            for feature in image_features:
                if (feature,category) not in self.log_likelihoods:
                    raise Exception("Feature not present in log-likelihoods.")
                
                log_probability += self.log_likelihoods[feature, category]
            
            probabilities.append(log_probability)

        return np.array(probabilities)
       
 
    #
    #   Classification Methods - the images must be given with pixels
    #   in range [0,1] (normalised)
    #

    # method to classify a single image given as a 28x28 numpy array
    def classify(self, image):
        # get probabilities of image belonging to each class
        probabilities = self.get_probabilities(image)
        max_probability = -1000000000
        prediction = -1
        for i in range(len(probabilities)):
            if probabilities[i] > max_probability:
                max_probability = probabilities[i]
                prediction = self.categories[i]
                
        return prediction
            
    
    # method to classify a given list of images
    def classify_images(self, images):
        return np.array([self.classify(image) for image in images])
    
    
    #
    #   Evaluation Methods
    #
    
    # method to evaluate the deployable NB model on the given dataset.
    # returns the accuracy of the model on the given dataset.
    def evaluate_on_dataset(self, dataset):
        nb_predictions = self.classify_images([img for img,_ in dataset])
        ground_truth = np.array([cat for _,cat in dataset])

        return np.sum(nb_predictions == ground_truth) / len(dataset)