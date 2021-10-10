#!/usr/bin/env python3

"""
Script to automate scoring and monitoring

Author: Megan McGee
Date: October 9, 2021
"""

import os
import json
import pickle
import pandas as pd
from sklearn.metrics import f1_score
import subprocess
import ingestion
import training
import scoring
import deployment
import diagnostics
import reporting


with open('config.json','r') as f:
    config = json.load(f)

input_folder_path = config['input_folder_path']
output_folder_path = config['output_folder_path']
model_path = config['output_model_path']
prod_deployment_path = os.path.join(config['prod_deployment_path'])

def run_full_process():
    # Check and read new data
    # first, read ingestedfiles.txt
    with open(os.path.join(os.getcwd(), output_folder_path, 'ingestedfiles.txt'), 'r') as f:
            ingested_files = f.read().split('\n')

    # second, determine whether the source data folder has files that aren't listed in ingestedfiles.txt
    new_files = []
    for f in os.listdir(os.path.join(os.getcwd(), input_folder_path)):
        if f not in ingested_files:
            new_files.append(f)


    # Deciding whether to proceed, part 1
    #if you found new data, you should proceed. otherwise, do end the process here
    if len(new_files) > 0:
        ingestion.merge_multiple_dataframe()
    else:
        return None


    # Checking for model drift
    # check whether the score from the deployed model is different from the score from the model that uses the newest ingested data
    with open(os.path.join(os.getcwd(), prod_deployment_path, 'latestscore.txt'), 'r') as f:
        latest_score = float(f.read())


    '''
    Note that scoring.py will not work here because it references a model in output_model_path
    and we need to reference prod_deployment_path, and it references data from test_data_path
    and we need to reference output_folder_path.

    I have written a new version for scoring below for the purpose here.
    '''

    # read in model from prod_deployment_path
    with open(os.path.join(os.getcwd(), prod_deployment_path,'trainedmodel.pkl'), 'rb') as f:
        current_model = pickle.load(f)

    # calculate the F1 score on the newly ingested data
    data = pd.read_csv(os.path.join(os.getcwd(), output_folder_path, 'finaldata.csv'))
    X = data[['lastmonth_activity', 'lastyear_activity', 'number_of_employees']]
    y = data['exited']
    predictions = current_model.predict(X)
    new_score = f1_score(y, predictions)

    print(new_score) ############# delete me

    # Deciding whether to proceed, part 2
    # if you found model drift, you should proceed. otherwise, do end the process here
    if new_score < latest_score:
        training.train_model()
        scoring.score_model()
    else:
        return None


    # Re-deployment
    # if you found evidence for model drift, re-run the deployment.py script
    deployment.store_model_into_pickle()

    # Diagnostics and reporting
    #run diagnostics.py and reporting.py for the re-deployed model
    reporting.score_model()
    subprocess.run(subprocess.run(["python3", "apicalls.py"]))

    return None


if __name__ == '__main__':
    run_full_process()
