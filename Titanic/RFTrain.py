"""
RFTrain.py
A random forest classifier to determine the survivors of the titanic.
"""
__author__ = "Copyright (C) 2016 Jared Levy"

from __future__ import print_function
from Imports import *
import RandomForestLib
import pandas
import numpy as np
import sys
import collections 

## uses code from random forest library

ML.Initialize()

read_test = pandas.read_csv ( "test.csv", quotechar='"' )
read_train = pandas.read_csv ( "train.csv", quotechar='"' )
num_examples = read_train.shape[0]
num_examples_test = read_test.shape[0]
## cabin is not needed (not always given) since it is synonymous with Pclass
## not all features given are needed...seven truly needed (name included)
num_features = 7

## pandas allows us to create a vector with all the names by searching "Name"
## note: names is a vector with type 'object' 
X = np.empty ( ( num_examples, num_features ) ) 
X = X.astype ( object )

X1 = np.empty ( ( read_test.shape[0], num_features ) )
X1 = X1.astype ( object )

## extracting the features from the training data
X[:,0] = read_train["Name"]
## fill in missing data
read_train["Pclass"].fillna('?')
X[:,1] = read_train["Pclass"].fillna('?')
X[:,2] = read_train["Sex"]
X[:,3] = read_train["SibSp"]
X[:,4] = read_train["Parch"]
read_train["Embarked"].fillna('?')
X[:,5] = read_train["Embarked"].fillna('?')
X[:,6] = read_train["Fare"]

## extracting the features from the test data
X1[:,0] = read_test["Name"]
read_test["Pclass"].fillna('?')
X1[:,1] = read_test["Pclass"].fillna('?')
X1[:,2] = read_test["Sex"]
X1[:,3] = read_test["SibSp"]
X1[:,4] = read_test["Parch"]
read_test["Embarked"].fillna('?')
X1[:,5] = read_test["Embarked"].fillna('?')
X1[:,6] = read_test["Fare"]

Y = np.empty ( ( num_examples ) )
## even though survived is binary, pandas still reads as object
Y = Y.astype ( object )
Y[:] = read_train["Survived"]



## build one classification tree
samples = np.arange ( num_examples )
oneTree = ML.BuildClassificationTree ( X, Y, samples, num_features )

prediction_test = np.zeros ( ( num_examples_test, 2 ) )
for i in range ( 0, num_examples_test ) :
  hypothesis_test = oneTree.predict ( X1[i,:] )
  prediction_test[i, 0] = i
  prediction_test[i, 1] = hypothesis_test

## generate a random forest
numberTrees = 300
num_features_forest = math.ceil ( math.sqrt ( float ( num_features ) ) )
forest = []
for specificTree in range ( 0, numberTrees ) :
  sys.stdout.write('.')
  sys.stdout.flush()
  ## initiate random sampling
  samples = np.random.choice ( num_examples, size=num_examples, replace=True )
  tree = ML.BuildClassificationTree ( X, Y, samples, num_features_forest )
  forest.append ( tree )

rf_prediction = np.zeros ( ( num_examples_test, 2 ) )  
for i in range ( 0, num_examples_test ) :
  rf_prediction[i, 0] = i
  h = 0.0
  counters = collections.defaultdict(int)
  for j in range ( 0, numberTrees ) :
    h = forest[j].predict ( X1[i,:] )
    counters[h] += 1
  counterMax = 0
  hMax = None
  for h, counter in counters.items() :
    if counter > counterMax :
      counterMax = counter
      hMax = h
      rf_prediction[i, 1] = hMax

## preparing for submission file
for i in range ( rf_prediction.shape[0] ) :
  rf_prediction[i, 0] = rf_prediction[i, 0] + 1

## create the submission file
submitFile = rf_prediction[:,1:]
outFile = open ( 'submit.csv', 'w' )
print ( 'PassengerId,Survived', file=outFile )
for i in range ( submitFile.shape[0] ) :
  print ( '%d,%d' % ( i + 892, submitFile[i] ), file=outFile )
outFile.close()
print ( ' ' )
print ( 'Done' )
