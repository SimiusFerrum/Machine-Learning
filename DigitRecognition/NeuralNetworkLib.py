__author__ = "Copyright (C) 2016 by Jared Levy"

from Imports import *

################################################################################

def InitializeRNGs ( seed ) :
  """ Initialize random number generators.
  """
  random.seed ( seed )
  numpy.random.seed ( seed )
  return None

################################################################################

def Initialize ( seed=12345 ) :
  """ Initialize various packages for use with this module.
  """
  # initialize random number generators
  InitializeRNGs ( seed )
  # turn-on interactive plotting
  matplotlib.pyplot.ion()
  return None

################################################################################

def Pause () :
  """ Pause for the user to press the enter key (e.g. after viewing a plot).
  """
  pause = raw_input ( "Program paused; press enter to continue" )
  return None

################################################################################

def NormalizationParameters ( X ) :
  """ Compute the parameters for mean normalization and feature scaling.
      Supports homogeneous data columns.
  """

  mu = numpy.mean ( X, axis=0, keepdims=True )
  sg = numpy.std  ( X, axis=0, keepdims=True )

  mu = mu.ravel()
  sg = sg.ravel()

  assert len(mu.shape) == 1
  assert len(sg.shape) == 1
  assert mu.shape == sg.shape

  assert \
    ( len(X.shape)==1 and mu.shape[0]==1 and sg.shape[0]==1 ) \
    or \
    ( len(X.shape)==2 and mu.shape[0]==X.shape[1] and sg.shape[0]==X.shape[1] )

  n = mu.shape[0]

  for i in range ( 0, n ) :
    if sg[i] == 0.0 :
      if mu[i] == 0.0 :
        sg[i] = 1.0
      else :
        sg[i] = mu[i]
        mu[i] = 0.0

  return mu, sg

################################################################################

def NormalizeOther ( X, mu, sg ) :
  """ Perform mean normalization and feature scaling given the parameters from
      some other matrix.
  """
  Xn = ( X - mu ) / sg
  assert Xn.shape == X.shape
  return Xn

################################################################################

def Normalize ( X ) :
  """ Perform mean normalization and feature scaling.
  """
  mu, sg = NormalizationParameters ( X )
  X = NormalizeOther ( X, mu, sg )
  return X, mu, sg

################################################################################

def PrependOnes ( X ) :
  """ Prepend a column of ones to a training data matrix for bias.
  """
  assert len(X.shape) == 2
  return numpy.column_stack ( ( numpy.ones ( X.shape[0], dtype=float ), X ) )

################################################################################

def InitializeWeights ( layers ) :
  """ Initialize neural network weights.
      Based on the paper "Improving the Learning Speed of 2-Layer Neural
      Networks by Choosing Initial Values of the Adaptive Weights" by
      Derrick Nguyen and Bernard Widrow, Stanford University.
  """
  nWeightMatrices = len(layers) - 1
  Wu = [None] * nWeightMatrices
  for i in range ( 0, nWeightMatrices ) :
    nLayerInputs  = layers[i] + 1 # with bias
    nLayerOutputs = layers[i+1]
    w = 0.7 * float(nLayerOutputs) ** (1.0/float(nLayerInputs))
    Wu[i] = numpy.random.rand ( nLayerOutputs, nLayerInputs ) * 2.0 * w - w
  return Wu

################################################################################

def PackWeights ( Wu ) :
  """ Pack a list of 2d neural network weight matrices into a 1d array.
  """
  return numpy.concatenate ( Wu, axis=None )

################################################################################

def UnpackWeights ( Wp, layers ) :
  """ Unpack a 1d array of neural network weights into a list of 2d matrices.
  """
  assert len(Wp.shape) == 1
  nWeightMatrices = len(layers) - 1
  Wu = [None] * nWeightMatrices
  beg = 0
  for i in range ( 0, nWeightMatrices ) :
    nLayerInputs  = layers[i] + 1 # with bias
    nLayerOutputs = layers[i+1]
    end = beg + nLayerInputs * nLayerOutputs
    Wu[i] = numpy.reshape ( Wp[beg:end], (nLayerOutputs,nLayerInputs) ) ;
    beg = end
  return Wu

################################################################################

def NeuralNetworkFwdProp ( X, Wu ) :
  """ Propagate input data forward through a neural network.
  """
  assert len(Wu) >= 1
  nLayers = len(Wu) + 1
  A = [None] * nLayers
  A[0] = X # must have a column of prepended ones!
  A[1] = Sigmoid ( A[0].dot(Wu[0].T) )
  for i in range ( 2, nLayers ) :
    A[i] = Sigmoid ( A[i-1].dot(Wu[i-1][:,1:].T) + Wu[i-1][:,0] )
  return A

################################################################################

def NeuralNetworkLogCost ( A, y ) :
  """ Compute the logistic cost for a neural network.
  """
  return - numpy.sum ( y*numpy.log(A) + (1.0-y)*numpy.log(1.0-A) )

################################################################################

def NeuralNetworkRegCost ( Lambda, Wu ) :
  """ Compute the regularization cost for a neural network.
  """
  if Lambda == 0.0 : return 0.0
  J = 0.0
  n = len(Wu)
  for i in range ( 0, n ) :
    J += numpy.sum ( numpy.inner(Wu[i],Wu[i]) )
  J *= Lambda * 0.5
  return J

################################################################################

def NeuralNetworkCostFunction ( Wp, X, y, layers, Lambda ) :
  """ Compute the cost and its gradients for a neural network.
  """
  Wu = UnpackWeights ( Wp, layers )
  A = NeuralNetworkFwdProp(X, Wu)
  D = NeuralNetworkBwdProp(A, y, Wu)
  m = X.shape[0]
  n = len(Wu)
  J = NeuralNetworkLogCost(A[n],y) + NeuralNetworkRegCost(Lambda,Wu)
  J /= float(m)
  Gu = [None] * n
  Gu[0] = numpy.einsum('ij,ik->jk',D[1],A[0]) + Lambda*Wu[0]
  for i in range ( 1, n ) :
    Ai = numpy.c_[X[:,0],A[i]] # X[:,0] is a column of ones
    Gu[i] = numpy.einsum('ij,ik->jk',D[i+1],Ai) + Lambda*Wu[i]
  Gp = PackWeights ( Gu )
  Gp /= float(m)
  return J, Gp

################################################################################

def Sigmoid ( z ) :
  """ Compute the sigmoid function for logistic regression and neural networks.
  """
  return 1.0 / ( 1.0 + numpy.exp(-z) )

################################################################################

def NeuralNetworkBwdProp ( A, y, Wu ) :
  """ Propagate output error backward through a neural network.
  """
  assert len(Wu) >= 1
  assert len(y.shape) == 2
  nLayers = len(Wu) + 1
  D = [None] * nLayers
  D[nLayers-1] = A[nLayers-1] - y
  for i in range ( nLayers-2, 0, -1 ) :
    D[i] = \
      numpy.take ( D[i+1].dot(Wu[i]), numpy.arange(1,Wu[i].shape[1]), axis=1 )
    D[i] *= A[i]
    D[i] *= ( 1.0 - A[i] )
  #D[0] not needed
  return D

################################################################################

