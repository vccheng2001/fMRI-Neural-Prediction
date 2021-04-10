# Gaussian Naive Bayes Classifier 
import pandas as pd 
import numpy as np
import math
from collections import Counter
'''
Gaussian Naive Bayes Classifier 
Given an fMRI image of 21764 voxels, predict the associated stimulus word/class
'''

name = "data"

def main():
    # number of voxels to select 
    k = 2000

    # load train data 
    train_dataset  = np.genfromtxt(f"data/train_{name}.csv", delimiter= ",", \
                                             skip_header=1, dtype='unicode')
    (means, stdevs, priors, best_k_voxels) = train_gnb(train_dataset, k)
    # Test
    test_dataset  = np.genfromtxt(f"data/test_{name}.csv", delimiter= ",", \
                                    skip_header=1, dtype='unicode')
    test_gnb(test_dataset, means, stdevs, priors, best_k_voxels)
   
        
def train_gnb(dataset, k):
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
    best_k_voxels = select_best_voxels(means, stdevs, k)    

    # calc P(Yi|X) for input image X 
    probs = calc_likelihoods(data, classes, priors, means, stdevs, best_k_voxels)

    # for each row, predict class yi
    preds = make_predictions(data, probs)            
    train_acc = eval_accuracy(preds, labels)

    print(f"Finished training, Train accuracy: {train_acc}")
    return (means, stdevs, priors, best_k_voxels)

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
def select_best_voxels(means, stdevs, k):
    diff = abs(means["tool"] - means["building"])
    best_k_voxels = np.argpartition(diff, -k)[-k:]
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
def make_predictions(data, probs, train=True):
    import itertools
    d = {k: probs[k] for k in list(probs.keys())[:5]}

    preds = []
    if train:
        out = open(f'predictions/train_{name}2000.txt', 'w')
    else:
        out = open(f'predictions/test_{name}2000.txt', 'w')

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
    return 1-err_rate

''' Test '''
def test_gnb(test_dataset, means, stdevs, priors, best_k_voxels):
    test_data   =   test_dataset[:,:-1]  # feature values 
    test_labels =   test_dataset[:,-1]   # last column (ground truth y)
    test_classes = split_data_by_class(test_dataset)
    
    # Calculate likelihoods given input test images 
    test_probs = calc_likelihoods(test_data, test_classes, priors, means, stdevs, best_k_voxels)
    
    # Make test predictions
    test_preds = make_predictions(test_data, test_probs, train=False)
    
    # Test accuracy 
    test_acc = eval_accuracy(test_preds, test_labels)
    print(f"Finished testing, Test accuracy: {test_acc}")
    return test_acc

if __name__ == "__main__":
    main()
