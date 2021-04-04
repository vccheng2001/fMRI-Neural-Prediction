# Gaussian Naive Bayes Classifier 
import pandas as pd 
import numpy as np
import math
from collections import Counter
'''
Given an fMRI image of 21764 voxels, predict the associated stimulus word/class
'''

def main():
    data_and_labels = np.genfromtxt("tiny_data.csv", delimiter=",", skip_header=1, dtype='unicode')
    
    # separate data, labels
    data   =   data_and_labels[:,:-1] # excludes label column
    labels =   data_and_labels[:,-1]  # label column

    num_rows, num_features = data.shape

    # Step 0: group training data by class, store into dict 
    classes = {}
    for row in data_and_labels:
        label = row[-1]
        data = row[:-1]
        if not classes.get(label): classes[label] = []
        classes[label].append(data)
    
    # Step 1: calc class priors P(Yi) for each class
    classes_count = Counter(labels) # dict of counts for each class 
    priors = {k: (v/num_rows) for k, v in classes_count.items()}

       # Step 2: calc P(Xi|Yi) for each feature-class pair
    ccp = class_cond_probs(data, classes, num_rows, num_features)
    
    # Step 3: calc P(Y|xi) for input image X 
    likelihoods = {}
    calc_likelihoods(likelihoods, data_and_labels, priors)
    
    # Step 4: choose class with max probability to make prediction
    

''' Calculates P(xi | yi) for each class-feature pair '''
def class_cond_probs(data, classes, features):
    ccp = {}
    for yi in classes:
        for i in range(1, num_features + 1):  # data_1 through data_200
            # store P(xi | yi)
            ccp[yi][xi] = gaussian_pdf(xi, mean, sigma)
    return ccp

''' Gaussian pdf '''
def gaussian_pdf(xi, mean, sigma):
    exp = math.exp( (- (xi - mean) **2) / (2 * (sigma**2)))
    return (1/math.sqrt(2*math.pi*sigma**2)) * exp

''' Calc likelihood that output belongs to a class yi given image xi 
    P(yi | xi) = P(x | yi) P(yi) '''
def calc_likelihoods(likelihoods, data_and_labels, priors):
    # iterate through feature matrix 
    # P(yi | xi) = P(x | yi) P(yi)
    for row in data_and_labels:
        yi = label
        x = data
        for xi in data:
            print(xi)
            prod *= ccp[yi][xi] 
    prod *= priors[yi]
    likelihoods[x][yi] = prod  




if __name__ == "__main__":
    main()

# def calc_mean(X, y):
#     # TODO 

# def calc_var(X, y):
    # TODO 