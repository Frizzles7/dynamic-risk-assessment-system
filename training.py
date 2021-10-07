#!/usr/bin/env python3

"""
Script to train a model and output pickle file of the model

Author: Megan McGee
Date: October 6, 2021
"""

from flask import Flask, session, jsonify, request
import pandas as pd
import numpy as np
import pickle
import os
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import json

# Load config.json and get path variables
with open('config.json','r') as f:
    config = json.load(f) 

dataset_csv_path = os.path.join(config['output_folder_path']) 
model_path = os.path.join(config['output_model_path']) 


# Function for training the model
def train_model():
    '''
    Ingest the data for training, split into features and target, train a
    logistic regression model, and output the trained model as a pickle file
    '''
    
    # ingest the data for training
    data = pd.read_csv(os.path.join(
        os.getcwd(),
        dataset_csv_path,
        'finaldata.csv'))

    # split data into features and target
    X = data[['lastmonth_activity', 'lastyear_activity', 'number_of_employees']]
    y = data['exited']

    # use this logistic regression for training
    clf = LogisticRegression(
        C=1.0,
        class_weight=None,
        dual=False,
        fit_intercept=True,
        intercept_scaling=1,
        l1_ratio=None,
        max_iter=100,
        multi_class='auto',
        n_jobs=None,
        penalty='l2',
        random_state=0,
        solver='liblinear',
        tol=0.0001,
        verbose=0,
        warm_start=False
        )
    
    # fit the logistic regression to your data
    clf.fit(X, y)
    
    # write the trained model to your workspace in a file called trainedmodel.pkl
    pickle.dump(clf, open(os.path.join(os.getcwd(), model_path,'trainedmodel.pkl'), 'wb'))


if __name__ == '__main__':
    train_model()
