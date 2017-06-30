from __future__ import print_function
from __future__ import division
from keras.layers import Dense, Dropout, Flatten
from keras.constraints import maxnorm
from keras.layers import Conv2D, MaxPooling2D
from keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import SGD
from keras.callbacks import ModelCheckpoint
from PIL import Image
from keras.models import Sequential
from keras import backend as K
from keras.models import load_model
import keras
import csv
import glob
import time
import math
import numpy as np
import os
import pickle
import scipy.ndimage
import scipy.signal
import skimage.feature
import skimage.morphology
import skimage.util
import struct
import sys
import time
import warnings
import csv
import pandas
import NeuralNetworkLib
import skimage.filters

################################################################################

def main () :
  neuralNetwork = load_model ( '/Users/Jared/MachineLearningJ/OutsideProjects/CatFeeder/model.hdf5' )
  
  augment_test_data = ImageDataGenerator ( rescale = 1./255 )
  test_data_generator = augment_test_data.flow_from_directory ( '/Users/Jared/MachineLearningJ/OutsideProjects/CatFeeder/test', target_size = ( 150, 150 ), batch_size = 20, class_mode = None, shuffle = False )

  predictions = neuralNetwork.predict_generator ( test_data_generator, steps = 12500 // 20, verbose = 1 )

  outFile = open ( 'submit.csv', 'w' )
  print ( 'id,label', file = outFile )
  for i in range ( predictions.shape[0] ) :
    for j in range ( 1 ) :
      print ( '%d,%f' % ( i + 1, predictions[i, j] ), file = outFile )
  outFile.close () 
  print ( 'Done' )  

################################################################################

if __name__ == '__main__' :
  main () 
