#!/usr/bin/env python3

"""
Script to generate confusion matrix plot for reporting

Author: Megan McGee
Date: October 7, 2021
"""

import pickle
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import json
import os
from diagnostics import model_predictions


# Load config.json and get path variables
with open('config.json','r') as f:
    config = json.load(f) 

test_data_path = os.path.join(config['test_data_path'])
model_path = os.path.join(config['output_model_path'])


# Function for reporting
def score_model():
    '''
    Use model_predictions from diagostics to calculate predictions.
    Prepare a confusion matrix plot and save as a png.
    '''

    # read in test data to get true values
    test_data = pd.read_csv(os.path.join(
        os.getcwd(),
        test_data_path,
        'testdata.csv'))
    y = test_data['exited']

    # calculate predictions
    predictions = model_predictions()

    # calculate the confusion matrix
    conf_matrix = confusion_matrix(y, predictions)

    # save confusion matrix
    cm_plot = plt.matshow(conf_matrix)
    cm_plot.figure.colorbar(cm_plot)
    cm_plot.axes.set_xlabel('Predicted')
    cm_plot.axes.set_ylabel('Actual')
    cm_plot.axes.set_title('Confusion Matrix')
    # add nicely formatted numbers to the plot
    # from https://stackoverflow.com/questions/20998083/show-the-values-in-the-grid-using-matplotlib
    for (i, j), z in np.ndenumerate(conf_matrix):
        cm_plot.axes.text(j, i, '{:0.1f}'.format(z), ha='center', va='center',
            bbox=dict(boxstyle='round', facecolor='white', edgecolor='0.3'))
    # save figure
    cm_plot.figure.savefig(os.path.join(os.getcwd(), model_path, 'confusionmatrix.png'))

    return None


if __name__ == '__main__':
    score_model()
