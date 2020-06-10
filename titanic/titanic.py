#######################################################################################################################
# titanic.py
# Author: Nolen Young
# Last modified: 6-9-2020
# Description: This file is the main file for my kaggle titanic competition solution. This includes simple driver code
# to take in command line arguments and run the program. The file imports data about
# titanic passengers and trains a machine learning model that will try to predict which passengers may have lived
# or perished. Link: https://www.kaggle.com/c/titanic/
#######################################################################################################################

import pandas as pd
import sys
import os
from sklearn.ensemble import RandomForestClassifier

features = ["Pclass", "Sex", "SibSp", "Parch"]

# Main
def main(trainFilePath, testFilePath):
    # read data from inFile into trainData
    try:
        trainData = pd.read_csv(trainFilePath)
        testData = pd.read_csv(testFilePath)
    except pd.errors.EmptyDataError as err:  # check for empty data set
        print("Error: {}".format(err))
        return 1

    model = learn(trainData)
    results = test(testData, model)

    output = pd.DataFrame({'PassengerId': testData.PassengerId, 'Survived': results})
    output.to_csv('results.csv', index=False)

    return 0


# runs the machine learning on
def learn(trainData):
    Y = trainData["Survived"]
    X = pd.get_dummies(trainData[features])
    model = RandomForestClassifier(n_estimators=1000, max_depth=50, random_state=1)
    return model.fit(X, Y)

def test(testData, model):
    X_test = pd.get_dummies(testData[features])
    return model.predict(X_test)


# parse command line args
if __name__ == "__main__":
    if len(sys.argv) == 3:  # must have two commands
        trainFilePath = sys.argv[1]
        testFilePath = sys.argv[2]

        if os.path.isfile(trainFilePath) and os.path.isfile(testFilePath): # check if file exists
            if not main(trainFilePath, testFilePath):  # RUN MAIN
                print("Done.")
                exit(0)
            else:
                print("titanic.py failed...\nDone.")
                exit(1)
        else:
            print("Error: File does not exists.")
            exit(1)
    else:
        print("Error: Invalid number of command line arguments. 3 arguments required.")
        exit(1)
else:
    print("Error: Improper usage of titanic.py")
    exit(1)
