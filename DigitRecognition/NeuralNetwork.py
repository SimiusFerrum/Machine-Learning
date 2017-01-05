"""
NeuralNetwork.py
A neural network (using my library) classifier to identify handwritten digits.
"""
__author__ = "Copyright (C) 2016 Jared Levy"

from __future__ import print_function
from __future__ import division
import csv
import glob
import math
import numpy as np
import os
import pickle
import scipy.ndimage
import scipy.signal
import skimage.feature
import skimage.filters
import skimage.morphology
import skimage.util
import struct
import sys
import time
import warnings
import csv
import pandas
import ML

################################################################################

def ShowProgressDot ( xk ) :
  print ( '.', end='' )
  sys.stdout.flush()

################################################################################

def main () :
  ## the train and test files were too large to upload to github
  ## can find on kaggle

  ## read the train file
  read = pandas.read_csv ( 'train.csv' )
  X2 = np.array ( read )
  ## seperate the image from the label
  ## X is the image array
  X = X2[:,1:]
  X = X.astype ( float )
  
  ## read the test file
  readTest = pandas.read_csv ( 'test.csv' )
  T = np.array ( readTest )
  T = T.astype ( float )
  
  ## Yknot is the label array (label of digit)
  Yknot = X2[:,0:1]
  ## Y is an array that has 0's for digits that aren't what's seen and 
  ## 1's for the classified digit
  Y = np.zeros ( ( 42000, 10 ) )
  for i in range ( Yknot.shape[0] ) :
    for j in range ( Yknot.shape[1] ) :
      Y[i, Yknot[i, j]] = 1

  ## initialize random number generators and interactive plotting
  ML.Initialize()
  
  ## get the number of training examples
  m = X.shape[0]
  ## get the number of features
  n = X.shape[1]

  ## perform mean normalization and feature scaling
  ## mu = average of each column of X
  ## sg = sigma (standard deviation) of each column of X
  X, mu, sg = ML.Normalize ( X )

  ## layers of the neural network
  nInputs = n
  nHidden = 28
  nOutput = 10
  layers = ( nInputs, nHidden, 14,  nOutput )
  lastLayerIndex = len ( layers ) - 1

  Wu = ML.InitializeWeights ( layers )
  Wp = ML.PackWeights ( Wu )
  
  ## prepend ones for bias
  X1 = ML.PrependOnes ( X )
  T1 = ML.PrependOnes ( T )
  n += 1

  Lambda = 0.0
  args = ( X1, Y, layers, Lambda )
  
  ## optimize and try the network
  maxiter = 1000
  print ( "%d iterations of conjugate-gradient descent:" % maxiter )
  result = scipy.optimize.minimize ( ML.NeuralNetworkCostFunction, Wp, args, 
    method="CG", jac=True, options={'maxiter':maxiter}, callback=ShowProgressDot  )
  print ( "\n" )
  Wp = result.x

  ## run the test file through the neural network
  Wu = ML.UnpackWeights ( Wp, layers )
  hypothesis = np.zeros ( ( T.shape[0], 10 ) )
  for i in range ( T.shape[0] ) :
    A = ML.NeuralNetworkFwdProp ( T1[i,:], Wu )
    for j in range ( 10 ) :
      if A[lastLayerIndex][j] == max ( A[lastLayerIndex] ) :
        hypothesis[i, j] = 1

  prediction = np.zeros ( ( T.shape[0], 1 ) )
  for i in range ( hypothesis.shape[0] ) :
     prediction[i] = np.argmax ( hypothesis[i,:] )

  ## create the submission file that contains the classifications
  submitFile = np.zeros ( ( T.shape[0], 1 ) )
  for i in range ( prediction.shape[0] ) :
    for j in range ( prediction.shape[1] ) :
      submitFile[i, j] = int ( prediction[i, j] )
  print ( submitFile )
  outFile = open ( 'submit.csv', 'w' ) 
  print ( 'ImageId,Label', file=outFile )
  for i in range ( submitFile.shape[0] ) :
    for j in range ( submitFile.shape[1] ) :
      print ( '%d,%d' % ( i + 1, submitFile[i, j] ), file=outFile )
  outFile.close()
  print ( 'Done' )  
  ## got ~85% accuracy
  
################################################################################

if __name__ == '__main__' :
  main()  
  
