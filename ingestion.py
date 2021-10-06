#!/usr/bin/env python3

"""
Script to ingest csv data files, combine into one dataframe,
and output result to a file

Author: Megan McGee
Date: October 5, 2021
"""

import pandas as pd
import numpy as np
import os
import json
from datetime import datetime


# Load config.json and get input and output paths
with open('config.json','r') as f:
    config = json.load(f) 

input_folder_path = config['input_folder_path']
output_folder_path = config['output_folder_path']


# Function for data ingestion
def merge_multiple_dataframe():
    '''
    Check for datasets in input folder, compile them together and de-deduplicate,
    write result to an output file, and write filenames ingested to text file.
    '''

    # setup empty dataframe to hold the data from the different files
    combined_df = pd.DataFrame(columns=['corporation',
                                        'lastmonth_activity',
                                        'lastyear_activity',
                                        'number_of_employees',
                                        'exited'])

    # setup empty list to store filenames ingested
    filenames = []

    # read in data from each file in the directory, and append each file's data to dataframe
    for f in os.listdir(os.path.join(os.getcwd(), input_folder_path)):
        temp_df = pd.read_csv(os.path.join(os.getcwd(), input_folder_path, f))
        combined_df = combined_df.append(temp_df, ignore_index=True)
        filenames.append(f)

    # drop any duplicates from the combined dataframe
    combined_df = combined_df.drop_duplicates()

    # output the dataframe to a csv file
    combined_df.to_csv(os.path.join(os.getcwd(), output_folder_path, 'finaldata.csv'), index=False)

    # output the list of filenames ingested to a text file
    with open(os.path.join(os.getcwd(), output_folder_path, 'ingestedfiles.txt'), 'w') as f:
        f.write('\n'.join(filenames))


if __name__ == '__main__':
    merge_multiple_dataframe()
