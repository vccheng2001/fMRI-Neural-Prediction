# Gaussian Naive Bayes Classifier 
import pandas as pd 
import numpy as np
import math
'''
Given an fMRI image of 21764 voxels, predict the associated stimulus word/class
'''
def main():
    print(gaussian(1.0,1.0,1.0))
    print(gaussian(2.0,1.0,1.0))
    print(gaussian(0.0,1.0,1.0))
 
def hi():
    data = pd.read_csv("tiny_data.csv", index_col = False)
    # training examples 
    # print(data.head())
    # first_col = data.iloc[:, 0]
    # print(first_col)
    num_rows     = data.shape[0]
    num_features = data.shape[1] - 1 # last column is label 
    # get list of unique classes 
    classes = data['label'].unique() 
    # Step 1: calc independent class probabilities
    priors = class_prior_probabilities(data, classes, num_rows)
    # Step 2: calc class-conditional probability for each feature-class pair 
    # class_cond_probs = class_cond_probabilities(data, classes, num_rows, num_features)
    # Step 3: calc likelihood P(X|Y) for input image X 
    likelihoods = calc_likelihoods(data, num_features)
    # choose class with max probability 

''' Calculates P(Y_i) for each class Y_i'''
def class_prior_probabilities(data, classes, num_rows):
    priors = {}
    for yi in classes:
        # number of rows where label == c
        count = len(data.loc[data['label'] == yi])
        # store prior probability for class c 
        priors[yi] = count / num_rows
    return priors

''' Calculates P(xi | yi) for each class-feature pair '''
def class_cond_probabilities(data, classes, num_rows, num_features):
    ccp = {}
    for yi in classes:
        for i in range(1, num_features + 1):  # data_1 through data_200
             # value of ith feature

            mean = 0
            sigma = 0
            # Gaussian distribution
            left  = 1/math.sqrt(2*math.pi*sigma**2)
            right = math.exp( (- (x_i - mean) **2) / (2 * (sigma**2)))
            # store P(xi | yi)
            ccp[yi][xi] = left * right 
    return ccp


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
