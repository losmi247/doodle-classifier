import numpy as np
from abc import ABC, abstractmethod


#
# Astract base class for a feature used by the 
# Naive Bayes classifier.
#

class NBFeature(ABC):    
    # method that takes a list of 28x28 pixels images,
    # and returns a dictionary that maps different
    # features OF SAME TYPE to the number of times they
    # appear in the set of images. Every implementation
    # of a feature must provide this method.
    @abstractmethod
    def count_feature_in_images(images):
        pass

    # method that takes a list of 28x28 pixels images,
    # and returns a dictionary that maps different
    # features OF ALL POSSIBLE TYPES to the number of times they
    # appear in the set of images.
    #
    # all implemented feature types must implement the count_feature_in_images
    # function, as indicated by the abstract method count_feature_in_images above.
    @staticmethod
    def count_features_in_images(images):
        feature_counts = {}
        # go through all implemented feature types
        for feature_class in NBFeature.all_feature_classes:
            feature_counts.update(feature_class.count_feature_in_images(images))
        return feature_counts

    # method to gather all features in a list, from all
    # extensions of this abstract base class.
    #
    # all implemented feature types must have a all_features static field
    # that contains a numpy array that contains all different features of that type.
    #
    # returns a numpy array of numpy arrays each of which contains
    # all different features of one type, and a tag saying which type of feature that is.
    @staticmethod
    def get_all_features():
        return np.array([np.insert(feature_class.all_features,0,feature_class) for feature_class in NBFeature.all_feature_classes])
    
    # method to extract all existing features in a given image.
    #
    # all implemented feature types must implement the extract_features
    # function to return a numpy array of features of that type from the image.
    @staticmethod
    def extract_features(image):
        return np.concatenate([feature_class.extract_features(image) for feature_class in NBFeature.all_feature_classes])


#
# A specific feature used by NB, pixel value at 
# a specific location - there are 28*28*256
# different possible such features.
#
class PositionedPixel(NBFeature):
    def __init__(self, x, y, value):
        self.x = x
        self.y = y
        self.value = value

    # override
    @staticmethod
    def count_feature_in_images(images):
        feature_frequency = {}
        for image in images:
            for x,y in np.ndindex(image.shape):
                feature = PositionedPixel.get_feature(x, y, image[x, y])
                if feature not in feature_frequency:
                    feature_frequency[feature] = 1
                else:
                    feature_frequency[feature] += 1
        return feature_frequency
    
    # method to encode a feature of type PositionedPixel as a unique number in range 0 to 28*28*256-1
    #
    # contract: if encode_feature(f1) == encode_feature(f2) for two features f1 and f2 of same type,
    # then f1 and f2 have the same values of all fields. 
    @staticmethod
    def encode_feature(feature):
        return feature.y*28*256 + feature.x*256 + feature.value
    
    # method to return a feature object with the given value and position
    @staticmethod
    def get_feature(x, y, value):
        return PositionedPixel.all_features[PositionedPixel.encode_feature(PositionedPixel(x, y, value))]
    
    # method to extract all existing PositionedPixel features
    # from the given image, in numpy array.
    #
    # all implementations of NBFeature must have a 'extract_features' method
    # that extracts all features from the array of all features of this type
    # (all_features) that appear in this given image.
    @staticmethod
    def extract_features(image):
        features = np.empty(28*28, NBFeature)
        for x,y in np.ndindex(image.shape):
            features[28*y+x] = PositionedPixel.get_feature(x, y, image[x, y])
        return features


#
# Static variables for the PositionedPixel class
#

# a numpy array of all possible features of this type
#
# all implementations of a NBFeature must have a static variable 'all_features'
# that is a numpy array of objects that represent each possible feature of that type.
#
# in addition, each implementation of NBFeature must have a function 'encode_feature(...)'
# that, given the values that uniquely define one feature of that type, returns its unique
# index (where it should be stored) in the 'all_features' array. also the 'get_feature'
# function that gets the feature with given values that define it (from the 'all_features'
# array).
PositionedPixel.all_features = np.empty(28*28*256, NBFeature)
for y in range(28):
    for x in range(28):
        for value in range(256):
            feature_object = PositionedPixel(x, y, value)
            PositionedPixel.all_features[PositionedPixel.encode_feature(feature_object)] \
                    = feature_object

          
#
# Static variables for the NBFeature class
#

# a list of implemented feature types, extend when implementing new
NBFeature.all_feature_classes = [PositionedPixel]