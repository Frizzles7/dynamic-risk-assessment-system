#!/usr/bin/env python3

"""
Script to call api endpoints

Author: Megan McGee
Date: October 9, 2021
"""

import requests
import os
import json

with open('config.json','r') as f:
    config = json.load(f)

model_path = os.path.join(config['output_model_path'])

# Specify a URL that resolves to your workspace
URL = "http://127.0.0.1:8000/"

# Call each API endpoint and store the responses
response1 = requests.get(URL + 'prediction?filepath=/testdata/testdata.csv').content
response2 = requests.get(URL + 'scoring').content
response3 = requests.get(URL + 'summarystats').content
response4 = requests.get(URL + 'diagnostics').content

# Combine all API responses
responses = str(response1) + '\n' + str(response2) + '\n' + str(response3) + '\n' + str(response4)

# Write the responses to your workspace
with open(os.path.join(os.getcwd(), model_path, 'apireturns.txt'), 'w') as f:
        f.write(responses)
