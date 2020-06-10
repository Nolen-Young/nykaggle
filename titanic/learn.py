#######################################################################################################################
# learn.py
# Author: Nolen Young
# Last modified: 6-9-2020
# Description: This file runs the learning algorithm on a given data set.
#######################################################################################################################
import pandas as pd
import numpy as np


# runs the machine learning on
def startLearning(inFilePath):
    # read data from inFile into trainingData
    try:
        trainingData = pd.read_csv(inFilePath)
    except pd.errors.EmptyDataError as err:  # check for empty data set
        print("Error: {}".format(err))
        return 1

    return 0
