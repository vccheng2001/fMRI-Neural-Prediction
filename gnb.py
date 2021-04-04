# Gaussian Naive Bayes Classifier 
import pandas as pd 

'''
Given an fMRI image of 21764 voxels, predict the associated stimulus word/class
'''
def main():
    data = pd.read_csv("tiny_data.csv", index_col = False)
    num_rows = len(data)
    print(data.head())
    # calc independent class probabilities
    priors = class_prior_probabilities(data, num_rows)
    # calc class-conditional probability for each feature-class pair 

    # calc likelihood P(X|Y) 

    # choose class with max probability 

def class_prior_probabilities(data, num_rows):
    priors = {}
    # get list of unique classes 
    classes = data['label'].unique()
    for c in classes:
        # number of rows where label == c
        count = len(data.loc[data['label'] == c]))
        # store prior probability for class c into dictionary
        priors[c] = count / num_rows
    return priors 


if __name__ == "__main__":
    main()
