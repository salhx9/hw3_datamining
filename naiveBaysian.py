#!bin/usr/python
# Shelby Luttrell

import pandas as pd
from sklearn.naive_bayes import GaussianNB
from sklearn import preprocessing
from sklearn.metrics import *

# Read in the csv of Q3 training data
q3train = pd.read_csv('q3_train.csv', sep= ',')

# set the categorical data to binary values 0 and 1
trainIn = pd.get_dummies(q3train.loc[:, 'Is_Home_or_Away':'Media'])
trainOut = q3train.Label

# Read in the csv of Q3 test data
data = pd.read_csv('q3_test.csv', sep=',')

# set the test categorical data to binary values
testIn = pd.get_dummies(data.loc[:, 'Is_Home_or_Away':'Media'])
testOut = data.Label

naiveBay = GaussianNB()
le = preprocessing.LabelEncoder()
le.fit(trainOut)
trainOut = le.transform(trainOut)
naiveBay.fit(trainIn, trainOut)

# make the prediction
prediction = naiveBay.predict(testIn)
le = preprocessing.LabelEncoder()
le.fit(testOut)
testOut = le.transform(testOut)
print(type(testOut))

# prints the prediction of win or lose as 1 or 0
print(prediction)

# prints classification_report to the screen
print(classification_report(testOut, prediction))
# end
