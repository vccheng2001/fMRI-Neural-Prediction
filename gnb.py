# Gaussian Naive Bayes Classifier 
import pandas as pd 
import numpy as np
import math
from collections import Counter
'''
Given an fMRI image of 21764 voxels, predict the associated stimulus word/class
'''

def gnb():
    data_and_labels = np.genfromtxt("sample_data.csv", delimiter=",", skip_header=1, dtype='unicode')
    
    # separate data, labels
    data   =   data_and_labels[:,:-1] # all except label column
    labels =   data_and_labels[:,-1]  # label column

    num_rows, num_features = data.shape # 200x200 for tiny_data

    ''' Step 0: group training data by class, store into dict '''
    classes = {}
    for row in data_and_labels:
        label = row[-1]
        data = row[:-1].astype(np.float)
        if not classes.get(label): classes[label] = []
        classes[label].append(data)
    

    ''' Step 1: calc class priors P(Yi) for each class '''
    counts = Counter(labels) # dict of counts for each class 
    priors = {k: (v/num_rows) for k, v in counts.items()}


    ''' Step 2: calc P(Xi|Yi) for each feature-class pair '''
    ccp =  class_cond_probs(classes, num_features)

    
    ''' Step 3: calc P(Yi|X) for input image X  ''' 
    # P(Yi|X) = P(X1|Yi)*P(X2|Yi)*.....P(Xm|Yi) * P(Yi)

    ''' Step 4: choose class with max probability to make prediction '''
    preds = make_predictions(data, likelihood)



''' Step 2: calc P(Xi|Yi) for each feature-class pair '''
def class_cond_probs(classes, num_features):
    means = {}
    stdevs = {}
    for yi, yi_rows in classes.items():
        # each key into dict: yi
        means[yi] = np.mean(yi_rows, axis = 0)
        stdevs[yi] = np.std(yi_rows, axis = 0)

    ccp = {}
    for yi in classes:
        for xj in range(num_features):
            ccp[yi][xj] = gaussian_pdf(xj, means[yi][xj], stdevs[yi][xj])
    return ccp

''' Gaussian pdf '''
def gaussian_pdf(xi, mean, stdev):
    exp = math.exp( (- (xi - mean) **2) / (2 * (stdev**2)))
    return (1/math.sqrt(2*math.pi*stdev**2)) * exp


''' Step 3: Calc likelihood that output belongs to a class yi given image X
    P(yi | x) = P(x | yi) P(yi) '''
def calc_likelihoods(classes):
    # P(yi | xi) = P(x | yi) P(yi)
    for yi in classes:
        yi = label
        x = data
        for xi in X:
            prod *= # P(xi | yi )  FILL IN 
    prod *= priors[yi] # multiply by prior to get joint prob 
    likelihoods[x][yi] = prod  # P(yi | xi) 


''' Step 4: for each row in training set, make class prediction '''
def make_predictions(data, likelihood):
    preds = {}
    for X in data: 
        preds[X] = max(likelihood[data], key=likelihood[data].get)
    return preds

if __name__ == "__main__":
    gnb()

