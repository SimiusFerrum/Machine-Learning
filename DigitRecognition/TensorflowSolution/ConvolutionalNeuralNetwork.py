"""
ConvolutionalNeuralNetwork.py
A CNN classifier using tensorflow to classify handwritten digits.
"""
__author__ = "Copyright (C) 2016 Jared Levy"

from __future__ import division
from __future__ import print_function
import tensorflow as tf
import pandas
import numpy as np
import csv

## train and test files were to large to upload to github
read = pandas.read_csv('train.csv')
testRead = pandas.read_csv('test.csv')
## data preperation
## read test file
T = np.array(testRead)
T = T.astype(float)
## get pixel intensity between 0 - 1 (originally between 0 - 255)
for i in range ( T.shape[0] ) :
  for j in range ( T.shape[1] ) :
    T[i, j] = T[i, j] / 255

## read training file
X2 = np.array(read)
## put image into a seperate array as labels
X = X2[:,1:]
X = X.astype(float)
for i in range ( X.shape[0] ) :
  for j in range ( X.shape[1] ) :
    X[i, j] = X[i, j] / 255

## Yknot is the label array
Yknot = X2[:,0:1]
## Y is the array with 0's for not and 1's for the classification
Y = np.zeros((42000, 10))
for i in range(Yknot.shape[0]) :
  for j in range(Yknot.shape[1]) :
    Y[i, Yknot[i, j]] = 1

####################################################

sess = tf.InteractiveSession()

## x is a placeholder for the inputted image arrays
x = tf.placeholder(tf.float32, shape=[None, 784])
## y_ is a placeholder for the targetted output classifications
y_ = tf.placeholder(tf.float32, shape=[None, 10])

sess.run(tf.initialize_all_variables())

def weight_variable(shape):
  initial = tf.truncated_normal(shape, stddev=0.1)
  return tf.Variable(initial)

def bias_variable(shape):
  initial = tf.constant(0.1, shape=shape)
  return tf.Variable(initial)

def conv2d(x, W):
  return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

def max_pool_2x2(x):
  return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
                        strides=[1, 2, 2, 1], padding='SAME')

## using a 5x5 patch
## with one input 
## and 32 features/outputs channels
W_conv1 = weight_variable([5, 5, 1, 32])
## create a bias for every feature/output channels
b_conv1 = bias_variable([32])

x_image = tf.reshape(x, [-1,28,28,1])

## we use the rectifier activation function
## convolve the 5x5 patch onto the image
## and account for the bias
h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)
## apply max pooling to discard useless information
h_pool1 = max_pool_2x2(h_conv1)

#########################Layer2#################################
 
W_conv2 = weight_variable([5, 5, 32, 64])
b_conv2 = bias_variable([64])

h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
h_pool2 = max_pool_2x2(h_conv2)

W_fc1 = weight_variable([7 * 7 * 64, 1024])
b_fc1 = bias_variable([1024])

h_pool2_flat = tf.reshape(h_pool2, [-1, 7*7*64])
h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)

keep_prob = tf.placeholder(tf.float32)
h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

W_fc2 = weight_variable([1024, 10])
b_fc2 = bias_variable([10])

y_conv=tf.nn.softmax(tf.matmul(h_fc1_drop, W_fc2) + b_fc2)

## cross entropy is the cost function 
cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y_conv), reduction_indices=[1]))
## minimize the cost function using the Adam Optimizer
train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)
correct_prediction = tf.equal(tf.argmax(y_conv,1), tf.argmax(y_,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
sess.run(tf.initialize_all_variables())

## train the neural network
nBatchConv = 50
en = 0
for i in range(200):
  be = en
  en = be + nBatchConv
  if en > X.shape[0] : en = X.shape[0]
  if i%100 == 0:
    train_accuracy = accuracy.eval(feed_dict={
	x: X[be: en, :], y_: Y[be: en, :], keep_prob: 1.0})    
    print("step %d, training accuracy %g"%(i, train_accuracy))
  train_step.run(feed_dict={x: X[be: en, :], y_: Y[be: en, :], keep_prob: 0.5})

## apply the neural network to the test data
hypothesis = sess.run ( y_conv, feed_dict = { x: T, keep_prob: 0.5 } )
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