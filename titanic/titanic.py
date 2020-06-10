#######################################################################################################################
# titanic.py
# Author: Nolen Young
# Last modified: 6-9-2020
# Description: This file is the main file for my kaggle titanic competition solution. The file imports data about
# titanic passengers and trains a machine learning model that will try to predict which passengers may have lived
# or perished. Link: https://www.kaggle.com/c/titanic/
#######################################################################################################################


import sys
import os

from learn import startLearning


# Main
def main(mode, inFilePath):
    if mode == '-learn':
        startLearning(inFilePath)
    elif mode == '-test':
        return 0
    else:
        print("Error: Invalid first argument, {}. First argument must be either -learn or -test.")
        return 1
    return 0


# parse command line args
if __name__ == "__main__":
    if len(sys.argv) == 3:  # must have two commands
        mode = sys.argv[1]
        inFilePath = sys.argv[2]

        if os.path.isfile(inFilePath): # check if file exists
            if not main(mode, inFilePath):  # RUN MAIN
                print("Done.")
                exit(0)
            else:
                print("titanic.py failed...\nDone.")
                exit(1)
        else:
            print("Error: File does not exists.")
            exit(1)
    else:
        print("Error: Invalid number of command line arguments. 2 arguments required.")
        exit(1)
else:
    print("Error: Improper usage of titanic.py")
    exit(1)
