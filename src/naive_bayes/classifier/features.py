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
    # After creating more features, update this method.
    @staticmethod
    def count_features_in_images(images):
        return PositionedPixel.count_feature_in_images(images)

    # method to gather all features in a list, from all
    # extensions of this abstract base class. To add more,
    # simply extend the returned list.
    @staticmethod
    def get_all_features():
        return PositionedPixel.all_features
    
    # method to extract all existing features in a given image.
    # After creating more features, update this method.
    @staticmethod
    def extract_features(image):
        return PositionedPixel.extract_features(image)


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
    # from the given iamge
    @staticmethod
    def extract_features(image):
        features = np.empty(28*28, NBFeature)
        for x,y in np.ndindex(image.shape):
            features[28*y+x] = PositionedPixel.get_feature(x, y, image[x, y])
        return features

# static variables for the PositionedPixel class, i.e. 
# a list of all possible features of this type
PositionedPixel.all_features = np.empty(28*28*256, NBFeature)
for y in range(28):
    for x in range(28):
        for value in range(256):
            PositionedPixel.all_features[PositionedPixel.encode_feature(x, y, value)] \
                    = PositionedPixel(x, y, value)