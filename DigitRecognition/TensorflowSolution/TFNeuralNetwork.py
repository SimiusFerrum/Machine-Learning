"""
TFNeuralNetwork.py
A neural network classifier using tensorflow to classify handwritten digits.
"""
__author__ = "Copyright (C) 2016 Jared Levy"

from __future__ import division
from __future__ import print_function
import tensorflow as tf
import pandas
import numpy as np
import csv

## train and test files were too large to upload to github
read = pandas.read_csv('train.csv')
testRead = pandas.read_csv('test.csv')
## data preperation
## read the test data into an array
T = np.array(testRead)
T = T.astype(float)
## make the pixel intensity between 0 - 1
for i in range ( T.shape[0] ) :
  for j in range ( T.shape[1] ) :
    T[i, j] = T[i, j] / 255
  
## put the image into an array (X)
X2 = np.array(read)
X = X2[:,1:]
X = X.astype(float)
for i in range ( X.shape[0] ) :
  for j in range ( X.shape[1] ) :
    X[i, j] = X[i, j] / 255
    
## put the labels into an array (Y)
Yknot = X2[:,0:1]
Y = np.zeros((42000, 10))
for i in range(Yknot.shape[0]) :
  for j in range(Yknot.shape[1]) :
    Y[i, Yknot[i, j]] = 1

##################################################

sess = tf.InteractiveSession()

## x is a placeholder for the inputted images
x = tf.placeholder(tf.float32, shape=[None, 784])
## y_ is a placeholder for the output classifications
y_ = tf.placeholder(tf.float32, shape=[None, 10])

## create weights for the 10 classes/outputs
## each image has 784 features
W = tf.Variable(tf.zeros([784,10]))
## create biases for each class
b = tf.Variable(tf.zeros([10]))

sess.run(tf.initialize_all_variables())

## softmax will compute the probablity for each class in an image
y = tf.nn.softmax(tf.matmul(x,W) + b)

## cross entropy is the cost function
cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y), reduction_indices=[1]))
##cross_entropy = -tf.reduce_sum(y_*tf.log(y + 1e-20))

## minimize the costfunction
train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)

## train the neural network using minibatches
nBatch = 100
e = 0
for i in range(1000):
  b = e
  e = b + nBatch
  sess.run ( train_step, feed_dict = { x: X[b: e, :], y_: Y[b: e, :]})

## run the test data through the neural network
hypothesis = sess.run ( y, feed_dict = { x: T } )
prediction = np.zeros ( ( T.shape[0], 1 ) )

for i in range ( hypothesis.shape[0] ) :
  for j in range ( hypothesis.shape[1] ) :
    if hypothesis[i, j] == hypothesis[i, np.argmax ( hypothesis[i,:] )] :
      prediction[i] = np.argmax ( hypothesis[i,:] )

## creating submission file
submitFile = np.zeros ( ( T.shape[0], 1 ) )
for i in range ( prediction.shape[0] ) :
  for j in range ( prediction.shape[1] ) :
    submitFile[i, j] = int( prediction[i, j] )
print ( submitFile )
outFile =  open ( 'submitNN.csv', 'w' )
print ( 'ImageId,Label', file=outFile )
for i in range ( submitFile.shape[0] ) :
  for j in range ( submitFile.shape[1] ) :
    print ( '%d,%d' % ( i+1, submitFile[i, j] ), file=outFile )
outFile.close()

## got ~ 91% accuracy
print ( 'Done' ) 
