# Gaussian Naive Bayes Classifier 
import pandas as pd 
import numpy as np
import math
from collections import Counter

'''
Gaussian Naive Bayes Classifier 
Given an fMRI image of 21764 voxels, predict the associated stimulus word/class
'''

def main():
    # Train 
    train_dataset  = np.genfromtxt("data/train_medium.csv", delimiter= ",", \
                                    skip_header=1, dtype='unicode')
    (means, stdevs) = train_gnb(train_dataset)

    # Test
    test_dataset  = np.genfromtxt("data/test_medium.csv", delimiter= ",", \
                                    skip_header=1, dtype='unicode')
    test_gnb(test_dataset, means, stdevs)


def train_gnb(dataset):
    # separate data, labels from training set 
    data   =   dataset[:,:-1]  
    labels =   dataset[:,-1]   # last column (ground truth y)
    # group training data by class 
    classes = split_data_by_class(dataset)
    # calc class priors P(Yi) for each class
    counts = Counter(labels) 
    priors = {k: (v/len(data)) for k, v in counts.items()}
    # calc mean, stdev for each feature 
    means, stdevs = calc_mean_stdev(classes)
    # calc P(Yi|X) for input image X 
    probs = calc_likelihoods(data, classes, priors, means, stdevs)
    # for each row, predict class yi
    preds = make_predictions(data, probs)
    err_rate = eval_error(preds, labels)

    print(f"Finished training, Train accuracy: {1-err_rate}")
    return (means, stdevs)

''' Group training data by class '''
def split_data_by_class(dataset):
    classes = {}
    for row in dataset:
        yi = row[-1]
        X = row[:-1].astype(np.float)
        if not classes.get(yi): classes[yi] = []
        # append row corresponding to class yi
        classes[yi].append(X)
    return classes 

''' Calculates mean, stdev for each feature'''
def calc_mean_stdev(classes):
    means = {}
    stdevs= {}
    for yi, yi_rows in classes.items():
        means[yi] = np.mean(yi_rows, axis = 0)
        stdevs[yi]= np.std(yi_rows, axis = 0)
    return means, stdevs

''' Gaussian probability distribution function '''
def gaussian_pdf(xi, mean, stdev):
    exp = math.exp( (-1* (xi - mean) **2) / (2 * (stdev**2)))
    return (1/math.sqrt(2*math.pi*stdev**2)) * exp

''' Calculate likelihood that output belongs to a class yi given image X '''
def calc_likelihoods(data, classes, priors, means, stdevs):
    probs = {}
    num_features = data.shape[1]

    # for each class
    for yi in classes: 
        # for each row in training data
        for i in range(len(data)):
            X = data[i]         # Row i of training set 
            prod = priors[yi]   # P(yi)
            if not probs.get(i): probs[i] = {}
            for j in range(num_features):
                # multiply by P(xi | yi) for each feature 
                prod  *= gaussian_pdf(float(X[j]), means[yi][j], stdevs[yi][j])
            probs[i][yi] = prod 
    return probs

''' For each row in dataset, make class prediction '''
def make_predictions(data, probs, train=True):
    preds = []
    if train:
        out = open('predictions/predictions_train.txt', 'w')
    else:
        out = open('predictions/predictions_test.txt', 'w')

    for i in range(len(data)):
        # Select class that yields max likelihood
        pred = max(probs[i], key=probs[i].get)
        preds.append(pred)
        out.write(str(preds[i]) + '\n')

    out.close()
    return preds

''' Evaluate error '''
def eval_error(preds, labels):
    errs = 0
    num_rows = len(labels)

    for i in range(num_rows):
        if preds[i] != labels[i]:
            errs += 1

    err_rate =  errs/num_rows
    return err_rate

''' Test '''
def test_gnb(test_dataset, means, stdevs):
    # separate data, labels from training set 
    test_data   =   test_dataset[:,:-1]  
    test_labels =   test_dataset[:,-1]   # last column (ground truth y)
    test_classes = split_data_by_class(test_dataset)
    test_counts = Counter(test_labels) 
    test_priors = {k: (v/len(test_data)) for k, v in test_counts.items()}
    # group training data by class 
    test_probs = calc_likelihoods(test_data, test_classes, test_priors, means, stdevs)
    test_preds = make_predictions(test_data, test_probs, train=False)
    test_err_rate = eval_error(test_preds, test_labels)
    print(f"Finished testing, Test accuracy: {1-test_err_rate}")


if __name__ == "__main__":
    main()
