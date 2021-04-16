# Gaussian Naive Bayes Classifier
import pandas as pd
import numpy as np
import math
from collections import Counter
import sys
import matplotlib.pyplot as plt
import imagesc as imagesc

'''
Gaussian Naive Bayes Classifier
Given an fMRI image of 21764 voxels, predict the associated stimulus word/class

python3 gnb.py train_data.csv test_data.csv train_out.labels test_out.labels
'''

def main():
    # load train data
    train_dataset  = np.genfromtxt(train_in, delimiter= ",", \
                                             skip_header=1, dtype='unicode')
    # load test data
    test_dataset  = np.genfromtxt(test_in, delimiter= ",", \
                                            skip_header=1, dtype='unicode')
    # train
    train_params = train_gnb(train_dataset, train_out, num_voxels)
    train_error = train_params[-1]
    # test 
    test_error =test_gnb(test_dataset, test_out, train_params)
    output_error(metrics_out, train_error, test_error)



# Output errors to metrics_out file
def output_error(metrics_out, train_error, test_error):
    with open(metrics_out, 'w') as metrics:
        train_string = "error(train): %f\n" % train_error
        test_string = "error(test): %f\n" % test_error
        metrics.write(train_string)
        metrics.write(test_string)
    metrics.close()

def train_gnb(dataset, train_out, num_voxels):
    # separate data, labels from training set
    data   =   dataset[:,:-1]    # feature values
    labels =   dataset[:,-1]     # last column (ground truth y)

    # group training data by class
    classes = split_data_by_class(dataset)

    # get class priors P(Yi) for each class
    counts = Counter(labels)
    priors = {k: (v/len(data)) for k, v in counts.items()}

    # calc mean, stdev for each feature
    num_features = data.shape[1]
    means, stdevs = calc_mean_stdev(classes, num_features)

    # select top k voxels
    best_k_voxels = select_best_voxels(means, stdevs, num_voxels)

    # calc P(Yi|X) for input image X
    probs = calc_likelihoods(data, classes, priors, means, stdevs, best_k_voxels)

    # for each row, predict class yi
    preds = make_predictions(data, probs, train_out)
    train_error = eval_accuracy(preds, labels)

    print(f"Finished training, Train accuracy: {1-train_error}")
    return (means, stdevs, priors, best_k_voxels, train_error)

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
def calc_mean_stdev(classes, num_features):
    means = {}
    stdevs= {}
    num_classes = len(classes)
    sum_stdevs = np.zeros(num_features)
    for yi, yi_rows in classes.items():
        means[yi] = np.mean(yi_rows, axis = 0)
        stdevs[yi]= np.std(yi_rows, axis = 0)
    return means, stdevs

''' Gaussian probability distribution function '''
def gaussian_pdf(xi, mean, stdev):
    exp = math.exp( (-1* (xi - mean) **2) / (2 * (stdev**2)))
    pdf = (1/(math.sqrt(2*math.pi*stdev**2))) * exp
    return pdf

''' Select top k features to use  '''
def select_best_voxels(means, stdevs, num_voxels):
    diff = abs(means["tool"] - means["building"])
    best_k_voxels = np.argpartition(diff, -num_voxels)[-num_voxels:]
    return best_k_voxels

''' Calculate likelihood that output belongs to a class yi given image X '''
def calc_likelihoods(data, classes, priors, means, stdevs, best_k_voxels):
    probs = {}
    num_rows, num_features = data.shape
    # for each row in training data
    for i in range(len(data)):
        # print(f"Row {i} of training set")
        X = data[i]         # Row i of training set
        # for each class
        for yi in classes:
            logsum = np.log(priors[yi])  # P(yi)
            if not probs.get(i): probs[i] = {}
            # Only use the best k features
            for j in best_k_voxels:
                # multiply by P(xi | yi) for each feature
                gpdf = gaussian_pdf(float(X[j]), means[yi][j], stdevs[yi][j])
                logsum +=  np.log(gpdf)
            probs[i][yi] = logsum

    return probs

''' For each row in dataset, make class prediction '''
def make_predictions(data, probs, out, train=True):
    import itertools
    d = {k: probs[k] for k in list(probs.keys())[:5]}

    preds = []
    out = open(out, 'w')

    for i in range(len(data)):
        # Select class that yields max likelihood
        pred = max(probs[i], key=probs[i].get)
        preds.append(pred)
        out.write(str(pred) + '\n')

    out.close()
    return preds

''' Evaluate error '''
def eval_accuracy(preds, labels):
    errs = 0
    num_rows = len(labels)

    for i in range(num_rows):
        if preds[i] != labels[i]:
            errs += 1

    err_rate =  errs/num_rows
    return err_rate

''' Test '''
def test_gnb(test_dataset, test_out, train_params):
    # Unpack params from trianing
    (means, stdevs, priors, best_k_voxels, train_error) = train_params

    test_data   =   test_dataset[:,:-1]  # feature values
    test_labels =   test_dataset[:,-1]   # last column (ground truth y)
    test_classes = split_data_by_class(test_dataset)

    # Calculate likelihoods given input test images
    test_probs = calc_likelihoods(test_data, test_classes, priors, means, stdevs, best_k_voxels)

    # Make test predictions
    test_preds = make_predictions(test_data, test_probs, test_out, train=False)

    # Test accuracy
    test_error = eval_accuracy(test_preds, test_labels)
    print(f"Finished testing, Test accuracy: {1-test_error}")
    return test_error

if __name__ == "__main__":
    (program, train_in, test_in, train_out, test_out, metrics_out, num_voxels) = sys.argv
    num_voxels = int(num_voxels)
    main()
