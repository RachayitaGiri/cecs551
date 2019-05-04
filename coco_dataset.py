# File to load the COCO dataset, or subset
import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

def load_data():
    '''
    Function to return the train and test sets: 80-20 split
    Output:
        X_train: (Ntr, 224x224x3)
        X_test:  (Nte, 224x224x3)
        y_train: (Ntr, 1)
        y_test:  (Nte, 1)
    where:
        Ntr = number of rows in the training set
        Nte = number of rows in the test set
    '''
    labels = pd.read_csv('/home/datasets/annotations/val_labels.csv', delimiter=',')
    X = pd.read_csv('/home/datasets/img2array.csv', delimiter=',')
    y = labels[:, 1:]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

    return (X_train, X_test, y_train, y_test)

def load_data_subset():
    '''
    Function to return the subset of train and test sets: 80-20 split
    Output:
        X_train: (Ntr, 224x224x3)
        X_test:  (Nte, 224x224x3)
        y_train: (Ntr, 1)
        y_test:  (Nte, 1)
    where:
        Ntr = number of rows in the training set
        Nte = number of rows in the test set
    '''
    labels = pd.read_csv('/home/datasets/annotations/val_labels.csv', delimiter=',', nrows=500)
    X = pd.read_csv('/home/datasets/img2array.csv', delimiter=',', nrows=500)
    y = labels[:, 1:]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

    return (X_train, X_test, y_train, y_test)
