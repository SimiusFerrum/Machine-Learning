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

def PrependOnes ( X ) :
  """ Prepend a column of ones to a training data matrix for bias.
  """
  assert len(X.shape) == 2
  return numpy.column_stack ( ( numpy.ones ( X.shape[0], dtype=float ), X ) )

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

def LinearRegressionCostFunction ( theta, X, y, Lambda ) :
  """ Compute regularized linear regression cost and its gradients.
  """

  # theta is a 1d vector of weights
  assert len(theta.shape) == 1

  # X is a 2d matrix of training examples (with a column of prepended ones)
  assert len(X.shape) == 2

  # y is a 1d vector of training labels
  assert len(y.shape) == 1

  # the number of training labels should match the number of training examples
  assert y.shape[0] == X.shape[0]

  # the number of weights should match the number of training features
  assert theta.shape[0] == X.shape[1]

  ## if xi is a specific example from a row of X, the hypothesis is:
  ## h(xi) = xi.theta

  ## quadratic cost and its gradient are used for linear regression
  ## (m is the number of training examples)
  ##
  ## J = (1/(2*m))*Sum_{i=1}^{m} [h(xi)-yi]^2
  ##   + (Lambda/(2*m)) tj^2 (j>0)
  ##
  ## G = (1/m)*Sum_{i=1}^{m} [h(xi)-yi]*xi
  ##   + (Lambda/m) tj (j>0)

  m = X.shape[0]
  d = X.dot(theta) - y
  J = numpy.inner(d,d) / float(2*m)
  J += (Lambda/float(2*m)) * numpy.inner(theta[1:],theta[1:])

  G = X.T.dot(d) / float(m)
  G[1:] += (Lambda/float(m)) * theta[1:]

  ## the gradient should have the same shape as the weights
  assert len(G.shape) == len(theta.shape)

  return J, G

################################################################################
