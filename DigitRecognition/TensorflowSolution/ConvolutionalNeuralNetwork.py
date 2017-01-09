"""
ConvolutionalNeuralNetwork.py
A CNN classifier using tensorflow to classify handwritten digits.
"""
__author__ = "Copyright (C) 2016 Jared Levy"

from __future__ import division
from __future__ import print_function
import tensorflow
import pandas
import numpy as np
import csv
import sys

## train and test files were to large to upload to github
read = pandas.read_csv ( 'train.csv' )
testRead = pandas.read_csv ( 'test.csv' )
## data preperation
## read test file
T = np.array ( testRead )
T = T.astype ( float )
## get pixel intensity between 0 - 1 (originally between 0 - 255)
for i in range ( T.shape[0] ) :
  for j in range ( T.shape[1] ) :
    T[i, j] = T[i, j] / 255

## read training file
X2 = np.array ( read )
## put image into a seperate array as labels
X = X2[:,1:]
X = X.astype ( float )
for i in range ( X.shape[0] ) :
  for j in range ( X.shape[1] ) :
    X[i, j] = X[i, j] / 255

## Yknot is the label array
Yknot = X2[:,0:1]
## Y is the array with 0's for not and 1's for the classification
Y = np.zeros ( ( 42000, 10 ) )
for i in range ( Yknot.shape[0] ) :
  for j in range ( Yknot.shape[1] ) :
    Y[i, Yknot[i, j]] = 1

####################################################

session = tensorflow.InteractiveSession()

## x is a placeholder for the inputted image arrays
x = tensorflow.placeholder ( tensorflow.float32, shape=[None, 784] )
## y is a placeholder for the output classifications (10)
y = tensorflow.placeholder ( tensorflow.float32, shape=[None, 10] )

session.run ( tensorflow.initialize_all_variables() )

def weight_variable ( shape ) :
  original = tensorflow.truncated_normal ( shape, stddev=0.1 )
  return tensorflow.Variable ( original )

def bias_variable ( shape ) :
  original = tensorflow.constant ( 0.1, shape=shape )
  return tensorflow.Variable ( original )

def convolve_2dimensions ( x, W ) :
  return tensorflow.nn.conv2d ( x, W, strides=[1, 1, 1, 1], padding='SAME' )

def max_pool_2x2 ( x ) :
  return tensorflow.nn.max_pool ( x, ksize=[1, 2, 2, 1],
                        strides=[1, 2, 2, 1], padding='SAME' )

## using a 5x5 patch
## with one input 
## and 32 features/outputs channels
Weights_layer1 = weight_variable ( [5, 5, 1, 32] )
## create a bias for every feature/output channels
bias_layer1 = bias_variable ( [32] )

x_image = tensorflow.reshape ( x, [-1,28,28,1] )

## we use the rectifier activation function
## convolve the 5x5 patch onto the image
## and account for the bias
convolve_layer1 = tensorflow.nn.relu ( convolve_2dimensions ( x_image, Weights_layer1 ) + bias_layer1 )
## apply max pooling to discard useless information
pooling_layer1 = max_pool_2x2 ( convolve_layer1 )

#########################Layer2#################################
 
Weights_layer2 = weight_variable ( [5, 5, 32, 64] )
bias_layer2 = bias_variable ( [64] )

convolve_layer2 = tensorflow.nn.relu ( convolve_2dimensions ( pooling_layer1, Weights_layer2 ) + bias_layer2 )
pooling_layer2 = max_pool_2x2 ( convolve_layer2 )

## feature construction of weights
Weights_feature_construction1 = weight_variable ( [7 * 7 * 64, 1024] )
bias_feature_construction1 = bias_variable ( [1024] )

flatten = tensorflow.reshape ( pooling_layer2, [-1, 7 * 7 * 64] )
layer2 = tensorflow.nn.relu ( tensorflow.matmul ( flatten, Weights_feature_construction1 ) + bias_feature_construction1 )

keep_prob = tensorflow.placeholder ( tensorflow.float32 )
## utilize dropout to avoid overfitting
dropout = tensorflow.nn.dropout ( layer2, keep_prob )

Weights_feature_construction2 = weight_variable ( [1024, 10] )
bias_feature_construction2 = bias_variable ( [10] )

softmax = tensorflow.nn.softmax ( tensorflow.matmul ( dropout, Weights_feature_construction2 ) + bias_feature_construction2 )
 
cost_function = tensorflow.reduce_mean ( -tensorflow.reduce_sum ( y * tensorflow.log ( softmax ), reduction_indices=[1] ) )
## minimize the cost function using the Adam Optimizer
train = tensorflow.train.AdamOptimizer( 1e-4 ).minimize ( cost_function )
session.run ( tensorflow.initialize_all_variables() )

## train the neural network
nBatchConv = 50
en = 0
for i in range(200):
  sys.stdout.write ( '.' )
  sys.stdout.flush()
  be = en
  en = be + nBatchConv
  if en > X.shape[0] : en = X.shape[0]
  train.run ( feed_dict={ x: X[be: en, :], y: Y[be: en, :], keep_prob: 0.5 } )

## propogate the test data through the cnn
hypothesis = session.run ( softmax, feed_dict = { x: T, keep_prob: 0.5 } )
prediction = np.zeros ( ( T.shape[0], 1 ) )

for i in range ( hypothesis.shape[0] ) :
  for j in range ( hypothesis.shape[1] ) :
    if hypothesis[i, j] == hypothesis[i, np.argmax ( hypothesis[i,:] )] :
      prediction[i] = np.argmax ( hypothesis[i,:] )

## Creating submission file with classifications
submitFile = np.zeros ( ( T.shape[0], 1 ) )
for i in range ( prediction.shape[0] ) :
  for j in range ( prediction.shape[1] ) :
    submitFile[i, j] = int ( prediction[i, j] )
print ( submitFile )
outFile =  open ( 'submitCNN.csv', 'w' )
print ( 'ImageId,Label', file=outFile )
for i in range ( submitFile.shape[0] ) :
  for j in range ( submitFile.shape[1] ) :
    print ( '%d,%d' % ( i+1, submitFile[i, j] ), file=outFile )
outFile.close()

print ( 'Done' )

## got ~ 92% accuracy
