# -*- coding: utf-8 -*-
"""
Created on Sun Apr 16 13:07:09 2017

@author: rickdberg

General purpose functions

"""
import numpy as np
from collections import defaultdict

# Average duplicates in concentration dataset, add seawater value, and make spline fit to first three values
# Averaging function (from http://stackoverflow.com/questions/4022465/average-the-duplicated-values-from-two-paired-lists-in-python)
def averages(names, values):
    # Group the items by name
    value_lists = defaultdict(list)
    for name, value in zip(names, values):
        value_lists[name].append(value)

    # Take the average of each list
    result = {}
    for name, values in value_lists.items():
        result[name] = sum(values) / float(len(values))

    # Make it a Numpy array and pull out values
    resultkeys = np.array(list(result.keys()))
    resultvalues = np.array(list(result.values()))
    sorted = np.column_stack((resultkeys[np.argsort(resultkeys)], resultvalues[np.argsort(resultkeys)]))
    return sorted

# Error calculated as relative root mean squared error of curve fit to reported values
def rmse(model_values, measured_values):
    return np.sqrt(((model_values-measured_values)**2).mean())

