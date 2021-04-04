# Gaussian Naive Bayes Classifier 
import pandas as pd 
import numpy as np
import math
from collections import Counter
'''
Given an fMRI image of 21764 voxels, predict the associated stimulus word/class
'''

def main():
    data = np.genfromtxt("tiny_data.csv", delimiter=",", skip_header=1, dtype='unicode')
    # tiny_data uses 200 features, 200 rows only
    num_rows = data.shape[0]
    num_features = data.shape[1] - 1 
    # separate features, labels
    features = data[:,:-1]
    labels =   data[:,-1]

    # Step 1: calc class priors P(Yi) for each class
    classes = Counter(labels)   # dict of class counts 
    priors = {k: (v/num_rows) for (k, v) in classes.items()}

    # Step 2: calc P(Xi|Yi) for each feature-class pair
    # class_cond_probs = class_cond_probs(data, classes, num_rows, num_features)
    # Step 3: calc P(X|Y) for input image X 
    likelihoods      = calc_likelihoods(data, num_features)
    # choose class with max probability to make prediction


''' Calculates P(xi | yi) for each class-feature pair '''
def class_cond_probs(data, classes, num_rows, num_features):
    ccp = {}
    for yi in classes:
        for i in range(1, num_features + 1):  # data_1 through data_200
            xi = 0
            # store P(xi|yi)
            ccp[yi][xi] = gaussian(xi, mean, sigma)
    return ccp

''' Gaussian pdf '''
def gaussian(xi, mean, sigma):
    exp = math.exp( (- (xi - mean) **2) / (2 * (sigma**2)))
    return (1/math.sqrt(2*math.pi*sigma**2)) * exp

def calc_likelihoods(data, num_features):
    for index, row in data.iterrows():
        features = row[:,-1]
        label = row[-1]
        print(features)
        exit(0)



if __name__ == "__main__":
    main()
