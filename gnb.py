# Gaussian Naive Bayes Classifier 
import pandas as pd 
import numpy as np
import math
from collections import Counter
'''
Given an fMRI image of 21764 voxels, predict the associated stimulus word/class
'''

def gnb():
    data_and_labels = np.genfromtxt("small_data.csv", delimiter=",", skip_header=1, dtype='unicode')
    
    # separate data from labels
    data   =   data_and_labels[:,:-1] 
    labels =   data_and_labels[:,-1]   # last column

    num_rows, num_features = data.shape # 200x200 for tiny_data

    # group training data by class, store into dictionary 
    classes = split_data_by_class(data_and_labels)

    # calc class priors P(Yi) for each class
    counts = Counter(labels) # dict of counts for each class 
    priors = {k: (v/num_rows) for k, v in counts.items()}

    # calc P(Xi|Yi) for each feature-class pair 
    means, stdevs = calc_mean_stdev(classes)
    # ccp =  class_cond_probs(classes, num_features, means_stdevs)
    
    # calc P(Yi|X) for input image X 
    # P(Yi|X) = P(X1|Yi)*P(X2|Yi)*.....P(Xm|Yi) * P(Yi)
    probs = calc_likelihoods(classes, data, num_features, priors, means, stdevs)

    ''' Step 4: choose class with max probability to make prediction '''
    preds = make_predictions(data, probs)

''' Step 0: Group training data by class, store into dictionary '''
def split_data_by_class(data_and_labels):
    classes = {}
    for row in data_and_labels:
        label = row[-1]
        data = row[:-1].astype(np.float)
        if not classes.get(label): classes[label] = []
        classes[label].append(data)
    return classes 

''' Step 2: calc P(Xi|Yi) for each feature-class pair '''
def calc_mean_stdev(classes):
    means = {}
    stdevs= {}
    for yi, yi_rows in classes.items():
        # each key into dict: yi
        means[yi] = np.mean(yi_rows, axis = 0)
        stdevs[yi]= np.std(yi_rows, axis = 0)
    return means, stdevs

''' Step 2: calc P(Xi|Yi) for each feature-class pair '''
# def class_cond_probs(classes, num_features, means, stdevs):
#     ccp = {}
#     for yi in classes:
#         if not ccp.get(yi): ccp[yi] = {}
#         for X in data_and_labels[:,:-1]:
#             for j in range(num_features):
#                 xj = 
#                 ccp[yi][xj] = gaussian_pdf(xj, means[yi][j], stdevs[yi][j])
#     return ccp


''' Gaussian pdf '''
def gaussian_pdf(xi, mean, stdev):
    exp = math.exp( (-1* (xi - mean) **2) / (2 * (stdev**2)))
    return (1/math.sqrt(2*math.pi*stdev**2)) * exp


''' Step 3: Calc likelihood that output belongs to a class yi given image X
    P(yi | x) = P(x | yi) P(yi) '''
def calc_likelihoods(classes, data, num_features, priors, means, stdevs):
    probs = {}
    for yi in classes: # for all classes 
        prod = priors[yi]
        # for each row in training data
        for i in range(len(data)):
            X = data[i]
            if not probs.get(i): probs[i] = {}
            for j in range(num_features):
                prod  *= gaussian_pdf(float(X[j]), means[yi][j], stdevs[yi][j])
            probs[i][yi] = prod
    return probs

''' Step 4: for each row in training set, make class prediction '''
def make_predictions(data, probs):
    preds = {}
    for i in range(len(data)):
        preds[i] = max(probs[i], key=probs[i].get)
    print("Predictions")
    print(preds)
    return preds

if __name__ == "__main__":
    gnb()

