#!/usr/bin/env python3

"""
Script to score a model on test data and output F1 score

Author: Megan McGee
Date: October 7, 2021
"""

from flask import Flask, session, jsonify, request
import pandas as pd
import numpy as np
import pickle
import os
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import json


# Load config.json and get path variables
with open('config.json','r') as f:
    config = json.load(f) 

test_data_path = os.path.join(config['test_data_path'])
model_path = os.path.join(config['output_model_path'])


# Function for model scoring
def score_model():
    '''
    Read in the test data, read in the trained model, calculate the F1 score
    on the test data, and write the score to a file
    '''

    # read in test data
    test_data = pd.read_csv(os.path.join(
        os.getcwd(),
        test_data_path,
        'testdata.csv'))

    # read in the trained model
    with open(os.path.join(os.getcwd(), model_path,'trainedmodel.pkl'), 'rb') as f:
        model = pickle.load(f)

    # calculate the F1 score on the test data
    X = test_data[['lastmonth_activity', 'lastyear_activity', 'number_of_employees']]
    y = test_data['exited']
    predictions = model.predict(X)
    score = f1_score(y, predictions)

    # write the score to a text file
    with open(os.path.join(os.getcwd(), model_path, 'latestscore.txt'), 'w') as f:
        f.write(str(score) + '\n')

    return score


if __name__ == '__main__':
    score_model()
