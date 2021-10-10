#!/usr/bin/env python3

"""
Script to setup api endpoints

Author: Megan McGee
Date: October 7, 2021
"""

from flask import Flask, session, jsonify, request
import pandas as pd
import numpy as np
import pickle
import json
import os
from diagnostics import model_predictions, dataframe_summary, missing_values, execution_time, outdated_packages_list
from scoring import score_model


# Set up variables for use in our script
app = Flask(__name__)
app.secret_key = '1652d576-484a-49fd-913a-6879acfa6ba4'

with open('config.json','r') as f:
    config = json.load(f) 

dataset_csv_path = os.path.join(config['output_folder_path']) 

prediction_model = None


# Prediction Endpoint
@app.route("/prediction", methods=['GET','POST','OPTIONS'])
def predict():        
    # call the prediction function from diagnostics.py
    my_file_path = request.args.get('filepath')
    preds = model_predictions(my_file_path)
    return str(preds)


# Scoring Endpoint
@app.route("/scoring", methods=['GET','OPTIONS'])
def scoring():
    # check the score of the deployed model
    score = score_model()
    return str(score)


# Summary Statistics Endpoint
@app.route("/summarystats", methods=['GET','OPTIONS'])
def summarystats():
    # check means, medians, and modes for each column
    summary  = dataframe_summary()
    return str(summary)


# Diagnostics Endpoint
@app.route("/diagnostics", methods=['GET','OPTIONS'])
def diagnostics():
    # check timing, percent NA values, and dependencies
    timings = execution_time()
    percent_na = missing_values()
    outdated = outdated_packages_list()
    return ' timings: ' + str(timings) + '\n percent na: ' + str(percent_na) + '\n outdated: ' + str(outdated)


if __name__ == "__main__":    
    app.run(host='0.0.0.0', port=8000, debug=True, threaded=True)
