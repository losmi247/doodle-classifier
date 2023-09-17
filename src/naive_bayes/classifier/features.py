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
    # that contains a numpy array of all different features of that type.
    @staticmethod
    def get_all_features():
        return np.concatenate([feature_class.all_features for feature_class in NBFeature.all_feature_classes])
    
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
    
    # method to encode a feature as a unique number in range 0 to 28*28*256-1
    @staticmethod
    def encode_feature(x, y, value):
        return y*28*256+x*256+value
    
    # method to return a feature object with the given value and position
    @staticmethod
    def get_feature(x, y, value):
        return PositionedPixel.all_features[PositionedPixel.encode_feature(x, y, value)]
    
    # method to extract all existing PositionedPixel features
    # from the given image, in numpy array.
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
PositionedPixel.all_features = np.empty(28*28*256, NBFeature)
for y in range(28):
    for x in range(28):
        for value in range(256):
            PositionedPixel.all_features[PositionedPixel.encode_feature(x, y, value)] \
                    = PositionedPixel(x, y, value)

          
#
# Static variables for the NBFeature class
#

# a list of implemented feature types, extend when implementing new
NBFeature.all_feature_classes = [PositionedPixel]