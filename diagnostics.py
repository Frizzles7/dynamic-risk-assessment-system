#!/usr/bin/env python3

"""
Script to perform diagnostics on data and model

Author: Megan McGee
Date: October 7, 2021
"""

import pandas as pd
import numpy as np
import timeit
import os
import json
import pickle
import subprocess

# Load config.json and get environment variables
with open('config.json','r') as f:
    config = json.load(f) 

dataset_csv_path = os.path.join(config['output_folder_path'])
test_data_path = os.path.join(config['test_data_path'])
model_path = os.path.join(config['output_model_path'])

# Function to get model predictions
def model_predictions(my_file_path):
    '''
    Read in the deployed model and test dataset, calculate and return a list of predictions
    '''
    # read in the given file
    test_data = pd.read_csv('.' + my_file_path)

    # read in the trained model
    with open(os.path.join(os.getcwd(), model_path,'trainedmodel.pkl'), 'rb') as f:
        model = pickle.load(f)

    # calculate the F1 score on the test data
    X = test_data[['lastmonth_activity', 'lastyear_activity', 'number_of_employees']]
    y = test_data['exited']
    predictions = model.predict(X)
    return list(predictions)


# Function to get summary statistics
def dataframe_summary():
    '''
    Read in finaldata.csv, calculate means, medians, and standard deviations for each numeric column, and
    output list of summary statistics
    '''

    # read in data file
    data = pd.read_csv(os.path.join(
        os.getcwd(),
        dataset_csv_path,
        'finaldata.csv'))

    # identify numeric columns
    numeric_columns = ['lastmonth_activity', 'lastyear_activity', 'number_of_employees', 'exited']

    # calculate summary statistics by column
    col_means = [data[col].mean() for col in numeric_columns]
    col_medians = [data[col].median() for col in numeric_columns]
    col_stdevs = [data[col].std() for col in numeric_columns]

    summary = [col_means, col_medians, col_stdevs]

    return summary


# Function to check for missing values
def missing_values():
    '''
    Read in finaldata.csv, count the number of missing values in each column,
    and return a list percent missing by column
    '''

    # read in data file
    data = pd.read_csv(os.path.join(
        os.getcwd(),
        dataset_csv_path,
        'finaldata.csv'))

    # count NA values by column
    count_na = [data[col].isna().sum() for col in data.columns]
    total = data.shape[0]
    percent_na = [c / total for c in count_na]

    return percent_na


# Function to get timings
def execution_time():
    '''
    Calculate timings of ingestion.py and training.py scripts
    '''

    # timing for ingestion.py
    ingest_start = timeit.default_timer()
    os.system('python3 ingestion.py')
    ingest_timing = timeit.default_timer() - ingest_start

    # timing for training.py
    training_start = timeit.default_timer()
    os.system('python3 training.py')
    training_timing = timeit.default_timer() - training_start

    return [ingest_timing, training_timing]


# Function to check dependencies
def outdated_packages_list():
    '''
    For each package, show currently installed version and most recent version
    '''
    outdated = subprocess.check_output(['pip', 'list', '--outdated'])

    return outdated


if __name__ == '__main__':
    print(model_predictions('/testdata/testdata.csv'))
    #dataframe_summary()
    #missing_values()
    #execution_time()
    #outdated_packages_list()
