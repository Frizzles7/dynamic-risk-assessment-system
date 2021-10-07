#!/usr/bin/env python3

"""
Script to copy the model files to deployment directory

Author: Megan McGee
Date: October 7, 2021
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
import shutil


# Load config.json and correct path variable
with open('config.json','r') as f:
    config = json.load(f) 

dataset_csv_path = os.path.join(config['output_folder_path'])
model_path = os.path.join(config['output_model_path'])
prod_deployment_path = os.path.join(config['prod_deployment_path']) 


# function for deployment
def store_model_into_pickle():
    '''
    Copy the model pickle file, the latest score text file, and the text file
    listing the data files ingested into the deployment directory
    '''

    # copy model pickle file
    shutil.copy(
        os.path.join(os.getcwd(), model_path,'trainedmodel.pkl'),
        os.path.join(os.getcwd(), prod_deployment_path,'trainedmodel.pkl')
        )

    # copy latest score text file
    shutil.copy(
        os.path.join(os.getcwd(), model_path,'latestscore.txt'),
        os.path.join(os.getcwd(), prod_deployment_path,'latestscore.txt')
        )

    # copy list of ingested data files
    shutil.copy(
        os.path.join(os.getcwd(), dataset_csv_path,'ingestedfiles.txt'),
        os.path.join(os.getcwd(), prod_deployment_path,'ingestedfiles.txt')
        )


if __name__ == '__main__':
    store_model_into_pickle()
